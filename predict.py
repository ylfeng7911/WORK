#用于预测

import cv2
from PIL import Image,ImageDraw,ImageFont
import numpy as np
import os
import time
import argparse
import torch
from torch import nn
import torchvision.transforms as T
# from main import get_args_parser as get_main_args_parser
from models import build_model
from models.deformable_detr import PostProcess
import datetime

torch.set_grad_enabled(False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] 当前使用{}做推断".format(device))



def get_args_parser():
    #GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/r50_deformable_detr.sh
    #tensorboard --logdir=./exps --port=6006
    
    #resume要考虑2个问题，1不确定要不要（要不要args.start_epoch != 0），2一定要（改一个bool值）
    #1.在resume自己的权重文件和resume官方预训练文件时需要改一下代码（后者不需要得到优化器和学习率调度器的信息）.
    #2.对于resume时学习率下降计数的重置。前者不要重置学习率初始化计数，后者要重置
    
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)        #2
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=180, type=int)           #50
    parser.add_argument('--lr_drop', default=160, type=int)          #40             
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')  #和上面的区别是：每隔多久衰减  /在第几回合衰减
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')#4 --> 5,加入尺度4x

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher 
    parser.add_argument('--set_cost_class', default=2, type=float,         
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_NWD', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients   每个损失的权重
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=1, type=float)  #2
    parser.add_argument('--NWD_loss_coef', default=3, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='/home/fengyulei/fengyulei_space/Data/MultiUAV-Dataset/TinyTinyCoco_format', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')


    now = datetime.datetime.now()
    time_str = now.strftime("%Y%m%d_%H%M%S")
    parser.add_argument('--output_dir', default=f'exps/{time_str}',                   #训练结束后手动改name
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')             #!!!!!
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default =False, action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser

# 图像数据处理
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_model(model_path , args):
    model, _, _ = build_model(args)
    # model.cuda()
    model.eval()
    state_dict = torch.load(model_path)  # <-----------修改加载模型的路径
    missing, unexpected = model.load_state_dict(state_dict["model"], strict=False)
    model.to(device)
    print("[INFO] load model success")
    return model

def plot_and_save(im, results, name, prob_threshold=0.3):
    draw = ImageDraw.Draw(im)
    scores = results[0]["scores"].detach().cpu().numpy()
    labels = results[0]["labels"].detach().cpu().numpy()
    boxes = results[0]["boxes"].detach().cpu().numpy()  # xyxy format
    for score, label, box in zip(scores, labels, boxes):#label = 1
        if score < prob_threshold:  # 可调置信度阈值
            continue
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        font = ImageFont.truetype("arial.ttf", size=16)
        # draw.text((x0, y0 - 20), f"{label}:{score:.2f}", font=font)
    im.save(f"predict_result/{name}.jpg")
    print(f"[INFO] save to ./predict_result/{name}.jpg !!!")
    return 

# 单张图像的推断
def detect(im, model, transform):
    '''
    1.预处理 2.得到COCO接口的结果（对应原图） 3.在原图上可视化BBOX
    '''
    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    
    #assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'
    
    img = transform(im).unsqueeze(0)
    img = img.to(device)
    outputs = model(img)    #字典
    orig_target_sizes=torch.tensor([[512, 640]], device='cuda:0')
    postprocessors = {'bbox':PostProcess()}
    results = postprocessors['bbox'](outputs, orig_target_sizes)        #coco格式的结果,长度=bs的列表,每个元素为字典.  
    return results


if __name__ == "__main__":
    
    main_args = get_args_parser().parse_args()
    #加载模型
    dfdetr = load_model('./exps/20250629_151241/checkpoint0179.pth',main_args)
    img_path = os.path.join("00002",'4778.jpg')
    im = Image.open(img_path)
    # with open("00002/1.jpg",'rb') as f:
    #     im = Image.open(f).convert("RGB")

    results = detect(im, dfdetr, transform)
    plot_and_save(im, results, name = '0001', prob_threshold=0.3)
    # print("[INFO] {} time: {} done!!!".format('00002.jpg', waste_time))



