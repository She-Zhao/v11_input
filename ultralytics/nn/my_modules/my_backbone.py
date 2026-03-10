import sys
import os
# 添加D:\Project\Multi_input\my_v11\ultralytics路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))   

# 基础库
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
if __name__ == '__main__':
    from my_attention import *
else:
    from .my_attention import *
# 官方自己的库
from ultralytics.nn.modules import (
    C2PSA,
    SPPF,
    C3k2,
    Conv
)

# ==============================================================================
# 基类 - 只负责提特征，不负责融合
# ==============================================================================

class PartialSharedBackbone(nn.Module):
    def __init__(self, c1, c2, w, N):
        super(PartialSharedBackbone, self).__init__()
        self.w = w
        self.c1 = [c1] + c2[:-1]        # 第一个输入+不包括最后一层的输出，yaml文件写的是每一层的输出
        self.c2 = c2
        self.N = N
        
        # 定义第0层和第1层的Conv    第2层的C3k2  (各分支独立)
        self.conv_0 = nn.ModuleList([Conv(self.c1[0], self.c2[0], k=3, s=2) for _ in range(self.N)])
        self.conv_1 = nn.ModuleList([Conv(self.c1[1], self.c2[1], k=3, s=2) for _ in range(self.N)])
        self.C3k2_2 = nn.ModuleList([C3k2(self.c1[2], self.c2[2], 1, False, 0.25) for _ in range(self.N)])
        
        # 第3层的共享Conv  第4层的共享C3k2
        self.shared_conv_3 = Conv(self.c1[3], self.c2[3], k=3, s=2)
        self.shared_C3k2_4 = C3k2(self.c1[4], self.c2[4], 1, False, 0.25)

        # 第5层的共享Conv  第6层的共享C3k2
        self.shared_conv_5 = Conv(self.c1[5], self.c2[5], k=3, s=2)   
        self.shared_C3k2_6 = C3k2(self.c1[6], self.c2[6], 1, True)

        # 第7层的共享Conv  第8层的共享C3k2
        self.shared_conv_7 = Conv(self.c1[7], self.c2[7], k=3, s=2)       
        self.shared_C3k2_8 = C3k2(self.c1[8], self.c2[8], 1, True)

        # 第9层SPPF
        self.shared_SPPF_9 = SPPF(self.c1[9], self.c2[9], k=5)
        
        # 第10层C3k2
        self.shared_C2PSA_10 = C2PSA(self.c1[10], self.c2[10])

    def forward(self, x):
        if x.shape[1] != self.N * self.c1[0]:     # 输入x的维度！= 图像数量*单图通道数
            x = x.repeat(1, self.N * self.c1[0]//x.shape[1], 1, 1)
        B = x.shape[0]
        x_split = torch.split(x, split_size_or_sections=3, dim=1)

        # --- 独立分支部分 ---
        # 权重不同，必须用 for 循环分别计算
        x0_list = [conv(img) for conv, img in zip(self.conv_0, x_split)]         
        x1_list = [conv(img) for conv, img in zip(self.conv_1, x0_list)]         
        x2_list = [bottle(img) for bottle, img in zip(self.C3k2_2, x1_list)]     

        # <--- 核心提速修改点：进入共享主干前，把 N 个图折叠进 Batch 维度 --->
        # 1. 将 List 中的 N 个 (B, C, H, W) 堆叠成 (B, N, C, H, W)
        x2_stacked = torch.stack(x2_list, dim=1)
        # 2. 获取当前的特征维度
        _, _, C_feat, H_feat, W_feat = x2_stacked.shape
        # 3. 压扁成 (B*N, C, H, W)，实现物理内存连续
        x2_bn = x2_stacked.view(B * self.N, C_feat, H_feat, W_feat)

        # --- 共享主干部分 ---
        # 彻底告别 for 循环！一次性算完 B*N 张图，榨干 GPU 并发算力
        x3_bn = self.shared_conv_3(x2_bn)                                      
        x4_bn = self.shared_C3k2_4(x3_bn)                  
        x5_bn = self.shared_conv_5(x4_bn)                   
        x6_bn = self.shared_C3k2_6(x5_bn)                   
        
        x7_bn = self.shared_conv_7(x6_bn)                   
        x8_bn = self.shared_C3k2_8(x7_bn)                                 
        x9_bn = self.shared_SPPF_9(x8_bn)                   
        x10_bn = self.shared_C2PSA_10(x9_bn)                

        # <--- 逆向操作：用辅助函数将 (B*N, C, H, W) 重新解开成 包含 N 个 Tensor 的 List --->
        def split_to_list(tensor_bn):
            _, C_out, H_out, W_out = tensor_bn.shape
            # 还原成 (B, N, C, H, W)
            tensor_b_n = tensor_bn.view(B, self.N, C_out, H_out, W_out)
            # 沿着 N 的维度切片，返回 List，完美遵守外层接口协议
            return [tensor_b_n[:, i, ...] for i in range(self.N)]

        x4_list = split_to_list(x4_bn)
        x6_list = split_to_list(x6_bn)
        x10_list = split_to_list(x10_bn)

        return [x4_list, x6_list, x10_list]

class FullySharedBackbone(nn.Module):
    def __init__(self, c1, c2, w, N):
        super(FullySharedBackbone, self).__init__()
        self.w = w
        self.c1 = [c1] + c2[:-1]        # 第一个输入+不包括最后一层的输出，yaml文件写的是每一层的输出
        self.c2 = c2
        self.N = N
        
        self.shared_conv_0 = Conv(self.c1[0], self.c2[0], k=3, s=2)
        self.shared_conv_1 = Conv(self.c1[1], self.c2[1], k=3, s=2)
        self.shared_C3k2_2 = C3k2(self.c1[2], self.c2[2], 1, False, 0.25)
        
        self.shared_conv_3 = Conv(self.c1[3], self.c2[3], k=3, s=2)
        self.shared_C3k2_4 = C3k2(self.c1[4], self.c2[4], 1, False, 0.25)

        self.shared_conv_5 = Conv(self.c1[5], self.c2[5], k=3, s=2)
        self.shared_C3k2_6 = C3k2(self.c1[6], self.c2[6], 1, True)

        self.shared_conv_7 = Conv(self.c1[7], self.c2[7], k=3, s=2)
        self.shared_C3k2_8 = C3k2(self.c1[8], self.c2[8], 1, True)

        self.shared_SPPF_9 = SPPF(self.c1[9], self.c2[9], k=5)
        
        self.shared_C2PSA_10 = C2PSA(self.c1[10], self.c2[10])

    def forward(self, x):
        if x.shape[1] != self.N * self.c1[0]:     # 输入x的维度！= 图像数量*单图通道数
            x = x.repeat(1, self.N * self.c1[0]//x.shape[1], 1, 1)
                    
        # <--- 修改点：利用 view 压缩通道维度，彻底消灭 for 循环，让 N 张图在 Batch 维度并行推理榨干算力
        B, _, H, W = x.shape
        # 将 (B, N*3, H, W) 变形为 (B*N, 3, H, W)，保证内存连续性
        x_reshaped = x.contiguous().view(B * self.N, 3, H, W)

        # 所有的层变成极其清爽的单行计算
        x0 = self.shared_conv_0(x_reshaped)        
        x1 = self.shared_conv_1(x0)            
        x2 = self.shared_C3k2_2(x1)       
        x3 = self.shared_conv_3(x2)                                       
        x4 = self.shared_C3k2_4(x3)                  
        x5 = self.shared_conv_5(x4)            
        x6 = self.shared_C3k2_6(x5)               
        x7 = self.shared_conv_7(x6)       
        x8 = self.shared_C3k2_8(x7)                                
        x9 = self.shared_SPPF_9(x8)                  
        x10 = self.shared_C2PSA_10(x9)

        # <--- 修改点：计算完成后，用辅助函数将 (B*N, C, H, W) 重新解开成 包含 N 个 Tensor 的 List，遵守接口协议
        def split_to_list(tensor_bn):
            _, C_feat, H_feat, W_feat = tensor_bn.shape
            tensor_b_n = tensor_bn.view(B, self.N, C_feat, H_feat, W_feat)
            return [tensor_b_n[:, i, ...] for i in range(self.N)]

        x4_list = split_to_list(x4)
        x6_list = split_to_list(x6)
        x10_list = split_to_list(x10)

        return [x4_list, x6_list, x10_list]

class FullyIndependentBackbone(nn.Module):
    def __init__(self, c1, c2, w, N):
        super(FullyIndependentBackbone, self).__init__()
        self.w = w
        self.c1 = [c1] + c2[:-1]        # 第一个输入+不包括最后一层的输出，yaml文件写的是每一层的输出
        self.c2 = c2
        self.N = N
        
        # 定义第0层和第1层的Conv    第2层的C3k2  
        self.conv_0 = nn.ModuleList([Conv(self.c1[0], self.c2[0], k=3, s=2) for _ in range(self.N)])
        self.conv_1 = nn.ModuleList([Conv(self.c1[1], self.c2[1], k=3, s=2) for _ in range(self.N)])
        self.C3k2_2 = nn.ModuleList([C3k2(self.c1[2], self.c2[2], 1, False, 0.25) for _ in range(self.N)])
        
        # 第3层的Conv  第4层的C3k2
        self.conv_3 = nn.ModuleList([Conv(self.c1[3], self.c2[3], k=3, s=2) for _ in range(self.N)])
        self.C3k2_4 = nn.ModuleList([C3k2(self.c1[4], self.c2[4], 1, False, 0.25) for _ in range(self.N)])

        # 第5层的Conv  第6层的C3k2
        self.conv_5 = nn.ModuleList([Conv(self.c1[5], self.c2[5], k=3, s=2) for _ in range(self.N)])
        self.C3k2_6 = nn.ModuleList([C3k2(self.c1[6], self.c2[6], 1, True) for _ in range(self.N)])

        # 第7层的Conv  第8层的C3k2
        self.conv_7 = nn.ModuleList([Conv(self.c1[7], self.c2[7], k=3, s=2) for _ in range(self.N)])
        self.C3k2_8 = nn.ModuleList([C3k2(self.c1[8], self.c2[8], 1, True) for _ in range(self.N)])

        # 第9层SPPF
        self.SPPF_9 = nn.ModuleList([SPPF(self.c1[9], self.c2[9], k=5) for _ in range(self.N)])
        
        # 第10层C3k2
        self.C2PSA_10 =  nn.ModuleList([C2PSA(self.c1[10], self.c2[10]) for _ in range(self.N)])

    def forward(self, x):
        if x.shape[1] != self.N * self.c1[0]:     # 输入x的维度！= 图像数量*单图通道数
            x = x.repeat(1, self.N * self.c1[0]//x.shape[1], 1, 1)        
        x_split = torch.split(x, split_size_or_sections=3, dim=1)

        # <--- 修改点：统一所有局部变量的命名为 _list 结尾
        x0_list = [conv(img) for conv, img in zip(self.conv_0, x_split)]                   
        x1_list = [conv(img) for conv, img in zip(self.conv_1, x0_list)]                   
        x2_list = [bottle(img) for bottle, img in zip(self.C3k2_2, x1_list)]               
        x3_list = [conv(img) for conv, img in zip(self.conv_3, x2_list)]                   

        x4_list = [bottle(img) for bottle, img in zip(self.C3k2_4, x3_list)]               
        x5_list = [conv(img) for conv, img in zip(self.conv_5, x4_list)]                   
        x6_list = [bottle(img) for bottle, img in zip(self.C3k2_6, x5_list)]               
        
        x7_list = [conv(img) for conv, img in zip(self.conv_7, x6_list)]                   
        x8_list = [bottle(img) for bottle, img in zip(self.C3k2_8, x7_list)]               
        x9_list = [sppf(img) for sppf, img in zip(self.SPPF_9, x8_list)]                   
        
        x10_list = [c2psa(img) for c2psa, img in zip(self.C2PSA_10, x9_list)]              

        return [x4_list, x6_list, x10_list]

# ==============================================================================
# 特征增强 - 是否进行 CBAM
# ==============================================================================
class CBAMRefiner(nn.Module):
    def __init__(self, ch4, ch6, ch10):
        super(CBAMRefiner, self).__init__()
        self.cbam_4 = CBAM(ch4)  
        self.cbam_6 = CBAM(ch6) 
        self.cbam_10 = CBAM(ch10)       

    def forward(self, x4_list, x6_list, x10_list):
        # <--- 修改点：变量名规范化，输出的依然是带有增强特征的 list
        x4_refined_list = [self.cbam_4(img) for img in x4_list]
        x6_refined_list = [self.cbam_6(img) for img in x6_list]
        x10_refined_list = [self.cbam_10(img) for img in x10_list]        
        
        return x4_refined_list, x6_refined_list, x10_refined_list      # 返回格式保持 [List, List, List]
    
# ==============================================================================
# 特征融合 - 采用 Conv 还是 Max 进行特征融合
# ============================================================================== 
class MaxFusion(nn.Module):
    def __init__(self):
        super(MaxFusion, self).__init__()
    
    def forward(self, x4_list, x6_list, x10_list):
        # <--- 修改点：此处的输出是最终用于检测头的特征张量，统一加 _fused 后缀
        x4_fused, _ = torch.stack(x4_list, dim=1).max(dim=1)
        x6_fused, _ = torch.stack(x6_list, dim=1).max(dim=1)
        x10_fused, _ = torch.stack(x10_list, dim=1).max(dim=1)
        return x4_fused, x6_fused, x10_fused

class ConvFusion(nn.Module):
    def __init__(self, ch4, ch6, ch10, N):
        super(ConvFusion, self).__init__()
        self.conv1x1_4 = Conv(ch4*N, ch4)
        self.conv1x1_6 = Conv(ch6*N, ch6)
        self.conv1x1_10 = Conv(ch10*N, ch10)
        
    def forward(self, x4_list, x6_list, x10_list):
        # <--- 修改点：统一加 _fused 后缀
        x4_fused = self.conv1x1_4(torch.cat(x4_list, dim=1))
        x6_fused = self.conv1x1_6(torch.cat(x6_list, dim=1))
        x10_fused = self.conv1x1_10(torch.cat(x10_list, dim=1))

        return x4_fused, x6_fused, x10_fused

# ==============================================================================
# 实例化模块 - 根据传入的参数，组合上述模块
# ============================================================================== 
# task.py 中传入的参数：args = [c1, c2, width, num_of_input]
class MultiStreamBackbone(nn.Module):
    def __init__(
        self,
        c1 = 3,
        c2 = None,
        w = 1.0,
        N = 6,
        base_type = 'partial',
        fusion_type = 'max',
        use_cbam = True        
    ):
        super().__init__()
        if base_type == 'partial':
            self.backbone = PartialSharedBackbone(c1, c2, w, N)
        elif base_type == 'share':
            self.backbone = FullySharedBackbone(c1, c2, w, N)
        elif base_type == 'independent':
            self.backbone = FullyIndependentBackbone(c1, c2, w, N)
        else:
            raise ValueError(f"输入的主干网络不符合要求！可选的主干网络：`partial`、`share`、`independent`，当前为: {base_type}")
        
        if fusion_type == 'max':
            self.fusion = MaxFusion()
        elif fusion_type == 'conv':
            self.fusion = ConvFusion(c2[4], c2[6], c2[10], N)
        else:
            raise ValueError(f"输入的融合策略不符合要求！可选的融合策略：`max`、`conv`，当前为: {fusion_type}")
        
        self.use_cbam = use_cbam
        if self.use_cbam:
            self.cbam = CBAMRefiner(c2[4], c2[6], c2[10])
        
    
    def forward(self, x):
        # 第一步：获取原始特征 (返回的是列表组)
        features_list = self.backbone(x)
        
        # 第二步：特征增强 (接收列表，返回的还是列表)
        if self.use_cbam:
            features_list = self.cbam(*features_list)
        
        # 第三步：特征融合 (接收列表，把列表拍扁成最终的单一 Tensor)
        # <--- 修改点：提取出的最终特征加上 _fused 后缀
        x4_fused, x6_fused, x10_fused = self.fusion(*features_list)
        
        # 严格遵守 YOLO 检测头需要的 11 元素输出格式
        return [None, None, None, None, x4_fused, None, x6_fused, None, None, None, x10_fused]
