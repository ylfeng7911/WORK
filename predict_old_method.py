#用于预测

import cv2
from PIL import Image
import numpy as np
import os
import time
import argparse
import torch
from torch import nn
# from torchvision.models import resnet50
import torchvision.transforms as T
# from main import get_args_parser as get_main_args_parser
from models import build_model

torch.set_grad_enabled(False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] 当前使用{}做推断".format(device))


def get_args_parser():
    #GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/r50_deformable_detr.sh
    
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=60, type=int)           #50
    parser.add_argument('--lr_drop', default=50, type=int)          #40
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
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
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

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

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='/home/fengyulei/fengyulei_space/Data/MultiUAV-Dataset/TinyCoco_format', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='output',                   #训练结束后手动改name
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='/home/fengyulei/fengyulei_space/Deformable-DETR/exps/tiny_60/checkpoint0059.pth', help='resume from checkpoint')             #!!!!!
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default = True, action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser


# 图像数据处理
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 将xywh转xyxy
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


# 将0-1映射到图像
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu().numpy()
    b = b * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    return b


# plot box by opencv
def plot_result(pil_img, prob, boxes, save_name=None, imshow=False, imwrite=False):
    LABEL = ['drone']

    opencvImage = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    if len(prob) == 0:
        print("[INFO] NO box detect !!! ")
        if imwrite:
            if not os.path.exists("./result/pred_no"):
                os.makedirs("./result/pred_no")
            cv2.imwrite(os.path.join("./result/pred_no", save_name), opencvImage)
        return

    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes):
        cl = p.argmax()
        label_text = '{}: {}%'.format(LABEL[0], round(p[cl] * 100, 2))      #LABEL[cl]

        cv2.rectangle(opencvImage, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 0), 2)
        cv2.putText(opencvImage, label_text, (int(xmin) + 10, int(ymin) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0), 1)

    if imwrite:
        if not os.path.exists("./result/pred"):
            os.makedirs('./result/pred')
        cv2.imwrite('./result/pred/{}'.format(save_name), opencvImage)

def print_layer_weights(model, layer_name_substring):
    """
    打印模型中包含指定关键词的层的权重张量（仅限权重参数）
    """
    print(f"[INFO] 查找包含关键词 '{layer_name_substring}' 的层：")
    found = False
    for name, param in model.named_parameters():
        if layer_name_substring in name:
            found = True
            print(f"\n[FOUND] 层名: {name}")
            print(f"→ 权重张量形状: {param.shape}")
            print(f"→ 权重值示例: {param.flatten()[:10]}")
            print(f"→ 均值: {param.mean().item():.6f}, 最大值: {param.max().item():.6f}, 最小值: {param.min().item():.6f}")
            is_zero = torch.all(param == 0).item()
            print(f"→ 是否全为 0: {is_zero}")
    if not found:
        print("[WARN] 未找到包含该关键词的层，请检查拼写或使用更宽泛关键词。")

def load_model(model_path , args):

    model, _, _ = build_model(args)
    # model.cuda()
    model.eval()
    state_dict = torch.load(model_path)  # <-----------修改加载模型的路径
    missing, unexpected = model.load_state_dict(state_dict["model"], strict=False)
    # print_layer_weights(model, "transformer.decoder.layers.0.linear1.weight")
    model.to(device)
    print("load model success")
    return model

# 单张图像的推断
def detect(im, model, transform, prob_threshold=0.3):
    # mean-std normalize the input image (batch-size: 1)
    

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    
    #assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'
    
    img = transform(im).unsqueeze(0)
    img = img.to(device)
    start = time.time()
    outputs = model(img)
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]  #去掉最后一类（背景？）
    keep = probas.max(-1).values > prob_threshold
    end = time.time()
    probas = probas.cpu().detach().numpy()
    keep = keep.cpu().detach().numpy()
    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    
    
    return probas[keep], bboxes_scaled, end - start


if __name__ == "__main__":
    
    main_args = get_args_parser().parse_args()
    #加载模型
    dfdetr = load_model('/home/fengyulei/fengyulei_space/Deformable-DETR/exps/20250625_234136/checkpoint0179.pth',main_args)
    # dfdetr = load_model('/home/fengyulei/fengyulei_space/Deformable-DETR/weights/r50_deformable_detr-checkpoint.pth',main_args)
    
    cn = 0
    waste=0
    img_path = os.path.join("00002",'000005.jpg')
    im = Image.open(img_path)
    # with open("00002/1.jpg",'rb') as f:
    #     im = Image.open(f).convert("RGB")

    scores, boxes, waste_time = detect(im, dfdetr, transform)
    plot_result(im, scores, boxes, save_name='test00002.jpg', imshow=False, imwrite=True)
    print("{} [INFO] {} time: {} done!!!".format(cn,'00002.jpg', waste_time))

    cn+=1
    waste+=waste_time



