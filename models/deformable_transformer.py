# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
    
    
class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = BoltzformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = BoltzformerDecoder(decoder_layer, num_decoder_layers, nhead, num_feature_levels, return_intermediate_dec,
                                          hidden_dim=d_model, mask_dim=d_model)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))  #4x256

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def unstack_features(self, features: torch.Tensor, spatial_shapes: list):
        """
        将堆叠的特征图 [b, sum_hw, c] 拆分为列表，每个元素为 [b, hw, c]
        
        Args:
            features: 输入特征图，形状为 [batch_size, sum_hw, channels]
            spatial_shapes: 各特征图的原始空间尺寸列表，例如 [(H1, W1), (H2, W2), ...]
        
        Returns:
            List[torch.Tensor]: 拆分后的特征图列表，每个元素形状为 [batch_size, hw, channels]
        """
        # 计算每个特征图的 hw 值（H*W）
        hw_list = [H * W for (H, W) in spatial_shapes]
        
        # 按 hw 分段拆分特征图
        split_features = torch.split(features, hw_list, dim=1)
        
        return list(split_features)
    
    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder. srcs --> src_flatten   堆叠
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten_list = []
        spatial_shapes_list = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes_list.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)                        #b,c,h,w -> b,hw,c
            mask = mask.flatten(1)                                      #b,h,w -> b,hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)            #b,c,h,w -> b,hw,c
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)    #[b,hw,c] + [1,1,c]   self.level_embed[lvl]用来在拼接后标注层级信息,
            lvl_pos_embed_flatten_list.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)     #在hw这一维度连接 b,Sum_hw,c
        mask_flatten = torch.cat(mask_flatten, 1)           #b,sum_hw
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten_list, 1)     #b,sum_hw,c
        spatial_shapes = torch.as_tensor(spatial_shapes_list, dtype=torch.long, device=src_flatten.device)        #[[64, 80],[32, 40],[16, 20],[ 8, 10]]
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1])) #[[64, 80],[32, 40],[16, 20],[ 8, 10]] -> [0,5120,6400,6720]
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)     #有效区域比例   [[1, 1] ...]

        # encoder 编码器，输出和输入同尺寸
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        
        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt, tgt_mask = torch.split(query_embed, c, dim=1)   #分成两份，用于参考点（位置查询）和内容查询（用于预测的实例）
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)   # nq,256 --> 1,nq,256
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)   # nq,256 --> 1,nq,256   都加上bs维度
            tgt_mask = tgt_mask.unsqueeze(0).expand(bs, -1, -1) 
            reference_points = self.reference_points(query_embed).sigmoid()     #query  (学习)-->  参考点   [bs,nq,2]
            init_reference_out = reference_points

        # unstacked_memory = self.unstack_features(memory, spatial_shapes_list)        #[b,hw,c]
        # max_feature =   srcs[0]       #[b,c,h,w]  换成最大特征图？
    
        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory,spatial_shapes, level_start_index,
                                            valid_ratios, tgt_mask, query_embed, mask_flatten)

    
        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, None, None


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn       维度：256-> 1024-> 256
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention  ;特征图和位置编码相加src+pos；输入输出都是b,sum_hw,c
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device): #为每个尺度的特征图生成归一化的网格参考点坐标(参考点个数=像素点个数)
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),   #划分：[0.5, 64 - 0.5] 划分成64份;y方向构成等差数列
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))    #x方向构成等差数列
            ref_y = ref_y.reshape(-1)[None] /  (valid_ratios[:, None, lvl, 1] * H_)  #归一化,shape=[bs,hw]
            ref_x = ref_x.reshape(-1)[None] /  (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)   #合成坐标，shape = [bs,hw,2]
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)  #所有参考点拼一起，shape = [bs,sum_hw,2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points     #shape = [bs,sum_hw,lvls,2]

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)     #q=k=实例query      [1,nq,256]
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1) #普通注意力 [1,nq,256]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),             #可分解注意力   reference_points是可学习的
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)    #结果[1,nq,256]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None): #query的位置编码就是query参数组的前一半
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):#解码器层
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:   #走这里
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]   #坐标 * 有效范围比例（调整到有效范围）
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:    #True
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)    #返回所有层的输出和参考点

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,        #256
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,       #1024，detr中是2048
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,     #4
        dec_n_points=args.dec_n_points,         # 4
        enc_n_points=args.enc_n_points,         # 4
        two_stage=args.two_stage,               
        two_stage_num_proposals=args.num_queries)   #300 ，detr中是100



#====================================================================

class BoltzformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.k = -1e9
        
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, tgt_mask, attn_mask, src_padding_mask=None):        
        # attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False        #防止全1  
        if attn_mask.dtype == torch.float:
            attn_mask = attn_mask * self.k      #作为float类型时，会直接加到注意力得分上,再softmax
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)     #q=k=实例query      [1,nq,256]
        tgt2 = self.self_attn(q.transpose(0, 1), 
                              k.transpose(0, 1), 
                              tgt.transpose(0, 1),
                              attn_mask=attn_mask,
                              )[0].transpose(0, 1) #要求[nq,b,c]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),             #可分解注意力   reference_points是可学习的
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)    #结果[1,nq,256]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # ffn
        tgt = self.forward_ffn(tgt)
        
        
        # self attention
        q = k = self.with_pos_embed(tgt_mask, query_pos)     #q=k=实例query      [1,nq,256]
        tgt_mask2 = self.self_attn(q.transpose(0, 1), 
                              k.transpose(0, 1), 
                              tgt_mask.transpose(0, 1),
                              attn_mask=attn_mask,
                              )[0].transpose(0, 1) #要求[nq,b,c]
        tgt_mask = tgt_mask + self.dropout2(tgt_mask2)
        tgt_mask = self.norm2(tgt_mask)
        # cross attention
        tgt_mask2 = self.cross_attn(self.with_pos_embed(tgt_mask, query_pos),             #可分解注意力   reference_points是可学习的
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)    #结果[1,nq,256]
        tgt_mask = tgt_mask + self.dropout1(tgt_mask2)
        tgt_mask = self.norm1(tgt_mask)
        # ffn
        tgt_mask = self.forward_ffn(tgt_mask)
        
        return tgt,tgt_mask


class BoltzformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, num_heads=8, num_feature_levels = 4,return_intermediate=False, 
                 hidden_dim=256, mask_dim = 256):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.decoder_norm = nn.LayerNorm(hidden_dim, eps=1e-4)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)     
        self.num_heads = num_heads
        self.num_feature_levels = num_feature_levels
        self.boltzmann_sampling: dict = {"mask_threshold": 0.5,"do_boltzmann": True,"sample_ratio": 0.1,"base_temp": 1,}
        
    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,tgt_mask,
                query_pos=None, src_padding_mask=None): #query的位置编码就是query参数组的前一半
        output = tgt
        intermediate = []
        intermediate_reference_points = []
        attn_mask = self.boltzmann(output, layer_id=-1)
        for lid, layer in enumerate(self.layers):#解码器层
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:   #走这里
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]   #坐标 * 有效范围比例（调整到有效范围）
            
            output,tgt_mask = layer(output, query_pos, reference_points_input, src, src_spatial_shapes,
                                    src_level_start_index,tgt_mask, attn_mask, src_padding_mask)
            attn_mask = self.boltzmann(tgt_mask, layer_id=lid)
            if self.return_intermediate:    #True
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)    #返回所有层的输出和参考点
        return output, reference_points


    
    def boltzmann(self, tgt_mask, layer_id=-1):
        # Boltzman sampling on attention mask
        threshold = self.boltzmann_sampling["mask_threshold"]  # original threshold for masked attention
        do_boltzmann = self.boltzmann_sampling["do_boltzmann"]  # whether to do Boltzman sampling
        sample_ratio = self.boltzmann_sampling["sample_ratio"]  # number of iid samples as a ratio of total number of masked tokens
        base_temp = self.boltzmann_sampling["base_temp"]  # base temperature for Boltzman sampling
              
        decoder_output = self.decoder_norm(tgt_mask)      #[b,nq,c]
        mask_embed = self.mask_embed(decoder_output)        #[b,nq,c]
        attn_mask = torch.matmul(mask_embed, mask_embed.transpose(-1, -2)) / (mask_embed.shape[-1] ** 0.5)  #自交互 [b,nq,nq]
        
        attn_mask = (
            attn_mask.sigmoid() #归一化
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
        )       #[b, nq, nq] -> [b * n_head, nq, nq]          .detach()   

        if do_boltzmann:
            # probability of Boltzman sampling
            Temp = base_temp / (2 + layer_id)  # temperature decays with layer number (first layer from id -1)
            boltzmann_prob = torch.exp(attn_mask / Temp)
            boltzmann_prob = (boltzmann_prob * (attn_mask < threshold).float())  # remove unmasked regions
            boltzmann_prob = boltzmann_prob / (boltzmann_prob.sum(dim=-1, keepdim=True) + 1e-6)
            
            assert not torch.isnan(boltzmann_prob).any(), f"NaN detected in attn_mask in layer {layer_id} in 1"
            
            # sample from Boltzman distribution n times
            n_samples = int(attn_mask.shape[-1] * sample_ratio)  # number of iid samples on the tokens    HW/10
            masked_prob = (1 - boltzmann_prob) ** n_samples  # probability that each token is still masked after n iid samples
                       
            rand_tensor = torch.rand_like(boltzmann_prob)
            boltzmann_mask = 1.0 - torch.sigmoid((rand_tensor - masked_prob) * 100)
            attn_mask = (1.0 - torch.sigmoid((attn_mask - threshold) * 100)) * boltzmann_mask  # 近似 and 操作（可导）
        else:
            attn_mask = (attn_mask < threshold).bool()
        return  attn_mask
    

