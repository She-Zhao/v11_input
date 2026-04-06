"""
验证集上推理，调用官方的实现，和训练时候验证的效果完全一致
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
# =====================================================================
# 🛡️ 强制环境隔离锁：确保导入的是你魔改后的 ultralytics
PROJECT_ROOT = "/data/ZS/v11_input"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# =====================================================================

from ultralytics import YOLO

def generate_official_json():
    # 1. 加载你训练好的权重
    model = YOLO("/data/ZS/defect-vlm/output/yolo_weights/iter3_10_row3_0p1_ema0p01.pt")
    
    # 2. 直接调用 val 方法，开启 save_json
    print("🚀 正在使用官方评估逻辑生成高精度预测框...")
    metrics = model.val(
        data="/data/ZS/v11_input/ultralytics/cfg/datasets/paint_semi/row3.yaml",
        conf=0.001,       # 极限阈值保召回
        iou=0.6,          # 官方验证阈值
        max_det=3000,     # 防止低分框被截断
        save_json=True,   # 保存 COCO 格式的 JSON
        plots=True,       # 是否画图
        project="/data/ZS/defect-vlm/output/yolo_decect",        # 保存父目录
        name="temp"      # 保存的名字
    )

    # 2. 从 metrics 对象中提取 mAP 并保存到 txt
    # metrics.save_dir 就是这次生成的文件夹路径 (例如 .../yolo_decect/iter3_10_row3_ema0p01)
    log_path = os.path.join(metrics.save_dir, "detailed_metrics.txt")
    
    with open(log_path, 'w', encoding='utf-8') as f:
        # 打印对齐的表头
        header = f"{'Class':>15} {'Precision':>12} {'Recall':>12} {'mAP@50':>12} {'mAP@50-95':>12}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        
        # 打印总体指标 (all)
        # metrics.box.mp, mr, map50, map 分别是平均/整体指标
        all_line = f"{'all':>15} {metrics.box.mp:>12.3f} {metrics.box.mr:>12.3f} {metrics.box.map50:>12.3f} {metrics.box.map:>12.3f}"
        f.write(all_line + "\n")
        
        # 打印各类别的详细指标
        # metrics.box.p, r, ap50, maps 分别对应各个类别的数值数组
        for i, class_id in enumerate(metrics.box.ap_class_index):
            class_name = model.names[class_id]
            p = metrics.box.p[i]           # 单个类别的 Precision
            r = metrics.box.r[i]           # 单个类别的 Recall
            ap50 = metrics.box.ap50[i]     # 单个类别的 mAP50
            ap = metrics.box.maps[i]       # 单个类别的 mAP50-95
            
            line = f"{class_name:>15} {p:>12.3f} {r:>12.3f} {ap50:>12.3f} {ap:>12.3f}"
            f.write(line + "\n")
            
    print(f"\n✅ 详细表格已完美保存至: {log_path}")
    
    # 顺便在终端预览一下写进去的内容，方便你确认
    print("\n📄 保存的文件内容预览:")
    with open(log_path, 'r', encoding='utf-8') as f:
        print(f.read())

if __name__ == '__main__':
    generate_official_json()
