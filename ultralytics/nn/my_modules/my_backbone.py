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
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    WorldDetect,
    v10Detect,
)

def count_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    return total_params

class MultiStreamBackbone(nn.Module):
    """
        原始类，只支持单张图像输入。
    """
    def __init__(self, c1, c2, w, N):
        super(MultiStreamBackbone, self).__init__()
        
        self.w = w
        # self.c1 = [c1] + [int(self.w * c) for c in c2[:-1]]     # 第一个输入+不包括最后一层的输出，yaml文件写的是每一层的输出
        # self.c2 = [int(self.w * c) for c in c2]
        self.c1 = [c1] + c2[:-1]                                    # 第一个输入+不包括最后一层的输出，yaml文件写的是每一层的输出
        self.c2 = c2
        self.N = N
        # 定义每层的输入输出通道数
        # self.w = w
        # self.c1 = [3] + [int(self.w * c) for c in [64, 128, 256, 256, 512, 512, 512, 1024, 1024, 1024]]
        # self.c2 = [int(self.w * c) for c in [64, 128, 256, 256, 512, 512, 512, 1024, 1024, 1024, 1024]]
        # self.N = N
        # 定义第0层和第1层的卷积    第2层的C3k2  
        self.conv_0 = nn.ModuleList([Conv(self.c1[0], self.c2[0], k=3, s=2) for _ in range(self.N)])
        self.conv_1 = nn.ModuleList([Conv(self.c1[1], self.c2[1], k=3, s=2) for _ in range(self.N)])
        self.C3k2_2 = nn.ModuleList([C3k2(self.c1[2], self.c2[2], 1, False, 0.25) for _ in range(self.N)])
        
        # 第3层的共享卷积  第4层的共享C3k2
        self.shared_conv_3 = Conv(self.c1[3], self.c2[3], k=3, s=2)
        self.shared_C3k2_4 = C3k2(self.c1[4], self.c2[4], 1, False, 0.25)

        # 第5层的共享卷积  第6层的共享C3k2
        self.shared_conv_5 = Conv(self.c1[5], self.c2[5], k=3, s=2)   
        self.shared_C3k2_6 = C3k2(self.c1[6], self.c2[6], 1, True)

        # 第7层的共享卷积  第8层的共享C3k2
        self.shared_conv_7 = Conv(self.c1[7], self.c2[7], k=3, s=2)       
        self.shared_C3k2_8 = C3k2(self.c1[8], self.c2[8], 1, True)

        # 第9层SPPF
        self.shared_SPPF_9 = SPPF(self.c1[9], self.c2[9], k=5)
        
        # 第10层C3k2
        self.shared_C2PSA_10 = C2PSA(self.c1[10], self.c2[10])

    def forward(self, x):
        outputs = []        
        if x.shape[1] != self.N * self.c1[0]:                       # 输入x的维度！= 图像数量*单图通道数
            x = x.repeat(1, self.N * self.c1[0]//x.shape[1], 1, 1)
        x_split = torch.split(x, split_size_or_sections=3, dim=1)

        x0 = [conv(img) for conv, img in zip(self.conv_0, x_split)]         # 第0层，分别通过 conv0
        x1 = [conv(img) for conv, img in zip(self.conv_1, x0)]              # 第1层，分别通过 conv1
        x2 = [bottle(img) for bottle, img in zip(self.C3k2_2, x1)]          # 第2层，分别通过 C3k2
        x3 = [self.shared_conv_3(img) for img in x2]                        # 第3层，共享卷积                                      

        x4 = [self.shared_C3k2_4(img) for img in x3]                        # 第4层，共享 C3k2         
        x5 = [self.shared_conv_5(img) for img in x4]                        # 第5层，共享卷积
        x6 = [self.shared_C3k2_6(img) for img in x5]                        # 第6层，共享C3k2

        
        x7 = [self.shared_conv_7(img) for img in x6]                        # 第7层，共享卷积
        x8 = [self.shared_C3k2_8(img) for img in x7]                        # 第8层，共享C3k2                              
        x9 = [self.shared_SPPF_9(img) for img in x8]                        # 第9层，共享SPPF 
        
        x10 = [self.shared_C2PSA_10(img) for img in x9]                     # x10:[8, 3072, 10, 10]                                    
        
        outputs = [None, None, None, None, torch.cat(x4, dim=1), None, torch.cat(x6, dim=1), None, None, None, torch.cat(x10, dim=1)]
        return outputs  # 返回每一层的输出

class MultiStreamBackbone_conv(MultiStreamBackbone):
    # 主干的多输入通过两个1*1降维卷积和neck的维度对齐。
    def __init__(self, c1, c2, w, N):
        super(MultiStreamBackbone_conv, self).__init__(c1, c2, w, N)
        self.conv1x1 = Conv(self.c2[4]*self.N, self.c2[4], k=1)             # 4 6层降维
        self.conv1x1_10 = Conv(self.c2[10]*self.N, self.c2[10], k=1)        # 10层降维

    def forward(self, x):
        outputs = []        
        if x.shape[1] != self.N * self.c1[0]:                       # 输入x的维度！= 图像数量*单图通道数
            x = x.repeat(1, self.N * self.c1[0]//x.shape[1], 1, 1)
        x_split = torch.split(x, split_size_or_sections=3, dim=1)

        x0 = [conv(img) for conv, img in zip(self.conv_0, x_split)]         # 第0层，分别通过 conv0
        x1 = [conv(img) for conv, img in zip(self.conv_1, x0)]              # 第1层，分别通过 conv1
        x2 = [bottle(img) for bottle, img in zip(self.C3k2_2, x1)]          # 第2层，分别通过 C3k2
        x3 = [self.shared_conv_3(img) for img in x2]                        # 第3层，共享卷积                                      

        x4 = [self.shared_C3k2_4(img) for img in x3]                        # 第4层，共享 C3k2
        x4 = torch.cat(x4, dim=1)
        x4_conv = self.conv1x1(x4)            
           
        x5 = [self.shared_conv_5(img) for img in x4]                        # 第5层，共享卷积
    
        x6 = [self.shared_C3k2_6(img) for img in x5]                        # 第6层，共享C3k2
        x6 = torch.cat(x6, dim=1)
        x6_conv = self.conv1x1(x6)                                    # 第6层实际保存的
        
        x7 = [self.shared_conv_7(img) for img in x6]                        # 第7层，共享卷积
        x8 = [self.shared_C3k2_8(img) for img in x7]                        # 第8层，共享C3k2                              
        x9 = [self.shared_SPPF_9(img) for img in x8]                        # 第9层，共享SPPF 
        
        x10 = [self.shared_C2PSA_10(img) for img in x9]                     # x10:[8, 3072, 10, 10]  
        x10 = torch.cat(x10, dim=1)                                  
        x10_conv = self.conv1x1_10(x10)
        
        outputs = [None, None, None, None, x4_conv, None, x6_conv, None, None, None, x10_conv]
        return outputs  # 返回每一层的输出


class MultiStreamBackbone_max(MultiStreamBackbone):
    # 主干的多输入通过逐通道取max和neck的维度对齐。
    def __init__(self, c1, c2, w, N):
        super(MultiStreamBackbone_max, self).__init__(c1, c2, w, N)

    def forward(self, x):

        if x.shape[1] != self.N * self.c1[0]:     # 输入x的维度！= 图像数量*单图通道数
            x = x.repeat(1, self.N * self.c1[0]//x.shape[1], 1, 1)
        x_split = torch.split(x, split_size_or_sections=3, dim=1)   
 
        x0 = [conv(img) for conv, img in zip(self.conv_0, x_split)]     # 第0层，分别通过 conv0
        x1 = [conv(img) for conv, img in zip(self.conv_1, x0)]          # 第1层，分别通过 conv1    
        x2 = [bottle(img) for bottle, img in zip(self.C3k2_2, x1)]      # 第2层，分别通过 C3k2
        x3 = [self.shared_conv_3(img) for img in x2]                    # 第3层，共享conv                               

        x4 = [self.shared_C3k2_4(img) for img in x3]                    # 第4层，共享 C3k2
        x4 = torch.cat(x4, dim=1)
        x4_max, _ = x4.view(x4.shape[0], self.N, x4.shape[1]//self.N, x4.shape[2], x4.shape[3]).max(dim=1)    # 第4层实际保存的，先拼接再CBAM      
           
        x5 = [self.shared_conv_5(img) for img in x4]                    # 第5层，共享卷积
            
        x6 = [self.shared_C3k2_6(img) for img in x5]                    # 第6层，共享C3k2    
        x6 = torch.cat(x6, dim=1)               
        x6_max, _ = x6.view(x6.shape[0], self.N, x6.shape[1]//self.N, x6.shape[2],x6.shape[3]).max(dim=1)  

        x7 = [self.shared_conv_7(img) for img in x6]                    # 第7层，共享卷积
        x8 = [self.shared_C3k2_8(img) for img in x7]                    # 第8层，共享C3k2                              
        x9 = [self.shared_SPPF_9(img) for img in x8]                    # 第8层，共享SPPF
        
        x10 = [self.shared_C2PSA_10(img) for img in x9]         # x10:[8, 3072, 10, 10]    
        x10 = torch.cat(x10, dim=1)  

        x10_max, _ = x10.view(x10.shape[0], self.N, x10.shape[1]//self.N, x10.shape[2], x10.shape[3]).max(dim=1)  
  
        outputs = [None, None, None, None, x4_max, None, x6_max, None, None, None, x10_max]
        return outputs  # 返回每一层的输出
    
    
class MultiStreamBackbone_cbam_max(MultiStreamBackbone):
    # 每个图像先进行CBAM，再沿着图像方向max（其实相当于是max添加了cbam）
    def __init__(self, c1, c2, w, N):
        super(MultiStreamBackbone_cbam_max, self).__init__(c1, c2, w, N)
        
        self.cbam_apart = CBAM(c1 = self.c2[4])                         # 用于 256 通道的 CBAM
        self.cbam_apart_10 = CBAM(c1 = self.c2[10])                     # 用于 512 通道的 CBAM
         
        self.conv1x1 = Conv(self.c2[4]*self.N, self.c2[4], k=1)         # 降维用的 1x1 卷积          
        self.conv1x1_10 = Conv(self.c2[10]*self.N, self.c2[10], k=1)

    def forward(self, x):
        if x.shape[1] != self.N * self.c1[0]:     # 输入x的维度！= 图像数量*单图通道数
            x = x.repeat(1, self.N * self.c1[0]//x.shape[1], 1, 1)
        x_split = torch.split(x, split_size_or_sections=3, dim=1)

        x0 = [conv(img) for conv, img in zip(self.conv_0, x_split)]            
        x1 = [conv(img) for conv, img in zip(self.conv_1, x0)]
        x2 = [bottle(img) for bottle, img in zip(self.C3k2_2, x1)]
        x3 = [self.shared_conv_3(img) for img in x2]                                        

        x4 = [self.shared_C3k2_4(img) for img in x3]         
        x4_cbamN = torch.cat([self.cbam_apart(img) for img in x4] , dim=1)
        x4_max, _ = x4_cbamN.view(x4_cbamN.shape[0], self.N, x4_cbamN.shape[1]//self.N, x4_cbamN.shape[2], x4_cbamN.shape[3]).max(dim=1)  
        
        x5 = [self.shared_conv_5(img) for img in x4]
    
        x6 = [self.shared_C3k2_6(img) for img in x5]
        x6_cbamN = torch.cat([self.cbam_apart(img) for img in x6], dim=1)       # 先CBAM再拼接
        x6_max, _ = x6_cbamN.view(x6_cbamN.shape[0], self.N, x6_cbamN.shape[1]//self.N, x6_cbamN.shape[2], x6_cbamN.shape[3]).max(dim=1)                                  
        
        x7 = [self.shared_conv_7(img) for img in x6]
        x8 = [self.shared_C3k2_8(img) for img in x7]                             
        x9 = [self.shared_SPPF_9(img) for img in x8]
        
        x10 = [self.shared_C2PSA_10(img) for img in x9]                            # x10:[8, 3072, 10, 10]   
        x10_cbamN = torch.cat([self.cbam_apart_10(img) for img in x10], dim=1)     # 先CBAM再拼接
        x10_max, _ = x10_cbamN.view(x10_cbamN.shape[0], self.N, x10_cbamN.shape[1]//self.N, x10_cbamN.shape[2], x10_cbamN.shape[3]).max(dim=1) 
        
        outputs = [None, None, None, None, x4_max, None, x6_max, None, None, None, x10_max]
        return outputs  # 返回每一层的输出

# 示例
if __name__ == "__main__":
    # summary(MultiStreamBackbone(c1=18), input_size=(8, 3 * 6, 320, 320))
    model = MultiStreamBackbone_conv(N=1)
    count_model_parameters(model)                   # 计算并打印参数量
    input_tensor = torch.randn(8, 3, 320, 320)  # batch_size=8, N=6, 输入尺寸128x128
    outputs = model(input_tensor)
    for i, output in enumerate(outputs):
        if output == None:
            print(f"Layer {i} output:", output)
        elif isinstance(output,list):
            for o in output:
                print(o.shape)
        else:
            print(f"Layer {i} output shapes:{output.shape},output type:{type(output)}" )
    # for output in outputs[-1]:
    #     print(output.shape)
    
    
    
            

    