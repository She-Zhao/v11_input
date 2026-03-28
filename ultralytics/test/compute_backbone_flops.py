import sys
import os
import torch
from thop import profile

# 动态添加项目根目录到 sys.path，确保能够导入 ultralytics 和你自己的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from ultralytics.nn.my_modules.my_backbone import MultiStreamBackbone

def main():
    # =====================================================================
    # 核心配置区
    # =====================================================================
    # 模拟输入图像大小
    IMG_SIZE = 300 
    
    # 【非常重要】这里需要填入你 yaml 文件里 YOLO 每一层的实际输出通道数
    # 下方是 YOLOv11s (或 v8s) 常见的通道列表结构，请根据你的实际情况核对修改
    C2_LIST = [32, 64, 64, 128, 128, 256, 256, 256, 256, 512, 512]
    
    # 实验组合清单 (完全对应 Markdown 表格)
    experiments = [
        {"name": "单流基线网络", "N": 1, "c1": 3, "base_type": "independent", "fusion_type": "max", "use_cbam": False},
        {"name": "多通道输入 (早期融合)", "N": 1, "c1": 6, "base_type": "independent", "fusion_type": "max", "use_cbam": True},
        {"name": "完全独立", "N": 2, "c1": 3, "base_type": "independent", "fusion_type": "max", "use_cbam": False},
        {"name": "完全独立 + CBAM", "N": 2, "c1": 3, "base_type": "independent", "fusion_type": "max", "use_cbam": True},
        {"name": "完全共享", "N": 2, "c1": 3, "base_type": "share", "fusion_type": "max", "use_cbam": False},
        {"name": "完全共享 + CBAM", "N": 2, "c1": 3, "base_type": "share", "fusion_type": "max", "use_cbam": True},
        {"name": "部分共享 + Conv", "N": 2, "c1": 3, "base_type": "partial", "fusion_type": "conv", "use_cbam": False},
        {"name": "部分共享 + CBAM-Conv", "N": 2, "c1": 3, "base_type": "partial", "fusion_type": "conv", "use_cbam": True},
        {"name": "部分共享 + Max", "N": 2, "c1": 3, "base_type": "partial", "fusion_type": "max", "use_cbam": False},
        {"name": "部分共享 + CBAM-Max (Ours)", "N": 2, "c1": 3, "base_type": "partial", "fusion_type": "max", "use_cbam": True},
    ]

    print(f"| 模型配置 | Params (M) | FLOPs (B) |")
    print(f"| :--- | :---: | :---: |")

    for exp in experiments:
        # 实例化网络
        model = MultiStreamBackbone(
            c1=exp['c1'],
            c2=C2_LIST,
            w=1.0,
            N=exp['N'],
            base_type=exp['base_type'],
            fusion_type=exp['fusion_type'],
            use_cbam=exp['use_cbam']
        )
        model.eval()

        # 生成假输入张量 (Batch_Size=1, Channels=c1*N, H, W)
        dummy_input = torch.randn(1, exp['c1'] * exp['N'], IMG_SIZE, IMG_SIZE)

        try:
            # 计算 MACs (乘加累积操作) 和 Parameters
            macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
            
            # 换算单位：
            # 1 MAC ≈ 2 FLOPs (目标检测领域的标准习惯)
            # FLOPs = (macs * 2) / 10^9 = B (Billion)
            # Params = params / 10^6 = M (Million)
            flops_b = (macs * 2) / 1e9
            params_m = params / 1e6
            
            # 特殊处理“独立检测”这种系统级跑两遍的情况
            if exp['name'] == "单流基线网络":
                print(f"| {exp['name']} | {params_m:.2f} | {flops_b:.2f} |")
                # 独立检测取最大值：参数量不变（同一个模型），但需要推理两次，算力翻倍
                print(f"| 独立检测 (结果取最大值) | {params_m:.2f} | {flops_b * 2:.2f} |")
            else:
                print(f"| {exp['name']} | {params_m:.2f} | {flops_b:.2f} |")
                
        except Exception as e:
            print(f"| {exp['name']} | Error: {e} | - |")

if __name__ == "__main__":
    main()