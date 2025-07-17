#把COCO指标记录在TB上

def log_coco_metrics(coco_evaluator, writer, epoch):
    if coco_evaluator is None:
        return
    
    if "bbox" in coco_evaluator.coco_eval:
        stats = coco_evaluator.coco_eval["bbox"].stats
        
        # 记录 AP 指标
        writer.add_scalar("AP/IoU_0.5:0.95", stats[0], epoch)
        writer.add_scalar("AP/IoU_0.5", stats[1], epoch)
        writer.add_scalar("AP/IoU_0.75", stats[2], epoch)
        writer.add_scalar("AP/Small", stats[3], epoch)
        writer.add_scalar("AP/Medium", stats[4], epoch)
        # writer.add_scalar("AP/Large", stats[5], epoch)
        
        # 记录 AR 指标
        # writer.add_scalar("AR/MaxDets_1", stats[6], epoch)
        # writer.add_scalar("AR/MaxDets_10", stats[7], epoch)
        writer.add_scalar("AR/MaxDets_100", stats[8], epoch)
        writer.add_scalar("AR/Small", stats[9], epoch)
        writer.add_scalar("AR/Medium", stats[10], epoch)
        # writer.add_scalar("AR/Large", stats[11], epoch)
        
        # 可选：将关键指标合并到同一图表（方便对比）
        # writer.add_scalars("AP_Summary", {
        #     "AP@[0.5:0.95]": stats[0],
        #     "AP@0.5": stats[1],
        #     "AP@0.75": stats[2]
        # }, epoch)
    return