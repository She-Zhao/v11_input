import torch.nn as nn
class MultiStreamBackbone(nn.Module):
    def __init__(self, c1, c2, w, N):
        super(MultiStreamBackbone, self).__init__()
        
        self.w = w
        self.c1 = [c1] + [int(self.w * c) for c in c2[:-1]]     # 第一个输入+不包括最后一层的输出，yaml文件写的是每一层的输出
        self.c2 = [int(self.w * c) for c in c2]
        self.N = N

class MultiStreamBackbone_cbam(MultiStreamBackbone):
    # 只有cbam 没有conv 只适用于单张图像，因为Neck部分的维度只支持原来的维度
    def __init__(self, c1, c2, w, N):
        super(MultiStreamBackbone_cbam, self).__init__(c1, c2, w, N)

args = [3, [64, 128, 256, 256, 512, 512, 512, 1024, 1024, 1024, 1024], 0.5, 1]
m = MultiStreamBackbone_cbam(*args)

