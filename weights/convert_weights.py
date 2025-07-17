import torch
pretrained_weights= torch.load('r50_deformable_detr-checkpoint.pth')
 
 
num_class = 1   #类别数
#修改状态字典key=class_embed.4.weight & class_embed.4.bias  (中间的数字从0-5)


for i in range(6):
    pretrained_weights["model"][f"class_embed.{i}.weight"].resize_(num_class+1, 256)
    pretrained_weights["model"][f"class_embed.{i}.bias"].resize_(num_class+1)
# pretrained_weights['model']['query_embed.weight'].resize_(300, 512)
torch.save(pretrained_weights, "r50_%d.pth"%num_class)