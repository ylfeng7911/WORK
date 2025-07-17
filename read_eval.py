#读取eval文件夹中的coco评估结果
import torch
import os
from pathlib import Path

def load_coco_eval_results(eval_dir="/home/fengyulei/fengyulei_space/Deformable-DETR/exps/r50_deformable_detr/eval"):
    eval_dir = Path(eval_dir)
    if not eval_dir.exists():
        print("⚠️ 评估目录不存在。")
        return

    files = sorted(eval_dir.glob("*.pth"))
    if not files:
        print("⚠️ 没有找到 .pth 文件。")
        return

    for file in files:
        print(f"\n📂 读取文件: {file.name}")
        data = torch.load(file)

        if not isinstance(data, dict):
            print("❌ 内容不是字典格式，跳过。")
            continue

        # 打印常见的指标
        keys_of_interest = ['params', 'counts', 'date', 'precision', 'recall', 'scores']

        for k in keys_of_interest:
            if k in data:
                print(f"🔹 {k}: {type(data[k])}")
            else:
                print(f"❌ 缺少字段: {k}")

        # 如果包含 stats（常见结构）
        if "stats" in data:
            stats = data["stats"]
            print("\n📊 COCO 主要指标:")
            print(f"  AP (IoU=0.50:0.95): {stats[0]:.3f}")
            print(f"  AP@0.50           : {stats[1]:.3f}")
            print(f"  AP@0.75           : {stats[2]:.3f}")
            print(f"  AR (maxDets=1)    : {stats[6]:.3f}")
            print(f"  AR (maxDets=10)   : {stats[7]:.3f}")
        else:
            print("❌ 未找到 stats 字段，无法打印 mAP。")



load_coco_eval_results()