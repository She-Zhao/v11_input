import sys
import os
import torch
from torchinfo import summary

# 动态添加项目根目录到 sys.path，确保能够导入 ultralytics 和你自己的模块
# 假设 test_backbone.py 在 D:\Github\v11_input\ultralytics\test\ 下
# 项目根目录应该是 D:\Github\v11_input\
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# import pdb; pdb.set_trace()
from ultralytics.nn.my_modules.my_backbone import MultiStreamBackbone
import thop



def test_all_combinations():
    """自动化测试所有模型架构组合，并输出维度和参数量"""
    
    # ---------------- 1. 模拟超参数设定 ----------------
    B = 2               # Batch size
    N = 6               # 图像数量 (视角数量)
    c1 = 3              # 每张图像的通道数 (RGB)
    H, W = 320, 320     # 输入图像尺寸
    
    # 模拟 YOLOv8s 的 backbone 每一层的输出通道数 (共 11 层)
    # 这个列表必须包含至少 11 个元素，对应第 0~10 层的通道
    c2_mock = [32, 64, 64, 128, 128, 256, 256, 512, 512, 512, 512] 
    
    print("=" * 60)
    print("🚀 开始多流 YOLO 骨干网络全面自动化测试")
    print("=" * 60)
    print(f"🔹 输入张量形状: Batch={B}, Streams={N}, Channels={c1}, H={H}, W={W}")
    print(f"🔹 期望的总体 Input Tensor: (B, N*c1, H, W) -> ({B}, {N*c1}, {H}, {W})\n")

    # 构造模拟输入数据
    dummy_input = torch.randn(B, N * c1, H, W)

    # 需要遍历的组合
    base_types = ['partial', 'share', 'independent']
    fusion_types = ['max', 'conv']
    cbam_options = [False, True]

    passed_tests = 0
    total_tests = len(base_types) * len(fusion_types) * len(cbam_options)

    # ---------------- 2. 遍历测试 ----------------
    for base in base_types:
        for fusion in fusion_types:
            for use_cbam in cbam_options:
                config_name = f"[{base.upper()}] + [{fusion.upper()}] + [CBAM:{use_cbam}]"
                print(f"⏳ 测试配置: {config_name:<45}", end="")
                
                try:
                    # 实例化模型
                    model = MultiStreamBackbone(
                        c1=c1, 
                        c2=c2_mock, 
                        w=1.0, 
                        N=N, 
                        base_type=base, 
                        fusion_type=fusion, 
                        use_cbam=use_cbam
                    )
                    model.eval() # 切换到推理模式

                    # 执行前向传播
                    with torch.no_grad():
                        outputs = model(dummy_input)
                    
                    # 验证输出长度和关键层形状
                    assert len(outputs) == 11, "YOLO 主干网络必须返回 11 个元素的列表！"
                    assert outputs[4].shape == (B, c2_mock[4], H//8, W//8), "第4层特征图形状错误！"
                    assert outputs[6].shape == (B, c2_mock[6], H//16, W//16), "第6层特征图形状错误！"
                    assert outputs[10].shape == (B, c2_mock[10], H//32, W//32), "第10层特征图形状错误！"
                    
                    print("✅ Pass")
                    passed_tests += 1
                    
                except Exception as e:
                    print(f"❌ Failed! 报错信息: {e}")

    print("-" * 60)
    print(f"🎉 自动化测试完成! 成功率: {passed_tests}/{total_tests}\n")


def profile_best_model():
    """为你最可能使用的配置生成详细的计算报告"""
    print("=" * 60)
    print("📊 生成主力模型性能报告 (Partial + Max + CBAM)")
    print("=" * 60)
    
    B, N, c1, H, W = 1, 6, 3, 640, 640
    c2_mock = [32, 64, 64, 128, 128, 256, 256, 512, 512, 512, 512] 
    dummy_input = torch.randn(B, N * c1, H, W)
    
    # 初始化你的主力配置
    model = MultiStreamBackbone(
        c1=c1, c2=c2_mock, w=1.0, N=N, 
        base_type='partial', fusion_type='max', use_cbam=True
    )
    
    # 1. 计算参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔹 总可训练参数量 (Params): {total_params / 1e6:.3f} M")
    
    # 2. 计算 FLOPs (浮点运算次数)
    if thop is not None:
        flops, _ = thop.profile(model, inputs=(dummy_input,), verbose=False)
        print(f"🔹 总体计算量 (FLOPs):      {flops / 1e9:.3f} G")
    
    # 3. 输出网络层级细节结构图
    print("\n🔍 网络层级明细 (Torchinfo Summary):")
    # depth=4 可以看到 backbone -> partial -> list/split 等内部结构
    summary(model, input_size=(B, N * c1, H, W), depth=4, device="cpu", 
            col_names=("input_size", "output_size", "num_params"))


if __name__ == '__main__':
    # 1. 先跑全量兼容性测试
    test_all_combinations()
    
    # 2. 再输出主力模型的性能分析报告
    profile_best_model()