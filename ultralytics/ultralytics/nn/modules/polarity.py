# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Polarity-Aware Dual-Stream Fusion Module

极性感知双流融合模块，用于挖掘ODG图像中红绿通道的物理含义。
在ODG算法中，红色(R)代表像素变亮（正梯度），绿色(G)代表像素变暗（负梯度）。

创新点：
1. 将输入通道分离为红光通道（正极性）和绿光通道（负极性）两个独立分支
2. 设计Polarity Interaction Module计算正负极性之间的空间关联
3. 特征融合后送入YOLO Neck

Reference:
    针对ODG图像中正负梯度（极性）的物理含义进行特征提取和融合
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("PolarityInteractionModule", "DualStreamPolarityFusion", "PolarityFusion")


class PolarityInteractionModule(nn.Module):
    """极性交互模块 (Polarity Interaction Module)
    
    通过加减法和注意力机制计算正负极性之间的空间关联。
    例如：鱼的前缘往往是正极性，后缘往往是负极性。
    
    Attributes:
        channels (int): 输入通道数
        reduction (int): 通道缩减比例
        spatial_kernel (int): 空间卷积核大小
    """
    
    def __init__(self, channels: int, reduction: int = 4, spatial_kernel: int = 3):
        """初始化极性交互模块
        
        Args:
            channels (int): 输入通道数
            reduction (int): 通道缩减比例，用于注意力计算
            spatial_kernel (int): 空间卷积核大小
        """
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        # 通道注意力：学习正负极性的重要性
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * 2, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力：学习正负极性的空间关联
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2),
            nn.Sigmoid()
        )
        
        # 极性交互卷积：学习正负极性之间的相互作用
        self.interaction_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, pos_feat: torch.Tensor, neg_feat: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            pos_feat (torch.Tensor): 正极性特征 (红色通道) [B, C, H, W]
            neg_feat (torch.Tensor): 负极性特征 (绿色通道) [B, C, H, W]
            
        Returns:
            (torch.Tensor): 融合后的特征 [B, C, H, W]
        """
        B, C, H, W = pos_feat.shape
        
        # 1. 极性差异计算：正负极性的差异和相似性
        diff = pos_feat - neg_feat  # 差异：前缘vs后缘
        sum_feat = pos_feat + neg_feat  # 相似性：共同特征
        
        # 2. 拼接正负极性特征
        concat_feat = torch.cat([pos_feat, neg_feat], dim=1)  # [B, 2C, H, W]
        
        # 3. 通道注意力：学习正负极性的重要性权重
        channel_att = self.channel_attention(concat_feat)  # [B, 2C, 1, 1]
        pos_att, neg_att = channel_att.chunk(2, dim=1)
        weighted_pos = pos_feat * pos_att
        weighted_neg = neg_feat * neg_att
        
        # 4. 空间注意力：学习正负极性的空间关联
        # 使用差异和相似性的平均值作为空间注意力输入
        spatial_input = torch.cat([
            diff.mean(dim=1, keepdim=True),  # 差异的平均
            sum_feat.mean(dim=1, keepdim=True)  # 相似性的平均
        ], dim=1)  # [B, 2, H, W]
        spatial_att = self.spatial_attention(spatial_input)  # [B, 1, H, W]
        
        # 5. 加权融合
        weighted_feat = weighted_pos + weighted_neg
        weighted_feat = weighted_feat * spatial_att
        
        # 6. 极性交互：学习正负极性之间的相互作用
        interaction_input = torch.cat([weighted_pos, weighted_neg], dim=1)
        interaction_feat = self.interaction_conv(interaction_input)
        
        # 7. 残差连接和最终融合
        output = weighted_feat + interaction_feat + sum_feat
        
        return output


class DualStreamPolarityFusion(nn.Module):
    """双流极性融合模块
    
    将输入图像分离为红色（正极性）和绿色（负极性）两个独立分支，
    分别进行特征提取，然后通过极性交互模块融合。
    
    Attributes:
        in_channels (int): 输入通道数（通常是3，RGB）
        out_channels (int): 输出通道数
        pos_branch (nn.Module): 正极性分支（红色通道）
        neg_branch (nn.Module): 负极性分支（绿色通道）
        interaction (PolarityInteractionModule): 极性交互模块
        fusion_conv (nn.Module): 融合卷积层
    """
    
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int = None, 
                 g: int = 1, d: int = 1, act: bool = True):
        """初始化双流极性融合模块
        
        Args:
            c1 (int): 输入通道数（通常是3，RGB）
            c2 (int): 输出通道数
            k (int): 卷积核大小
            s (int): 步长
            p (int, optional): 填充
            g (int): 分组数
            d (int): 膨胀率
            act (bool): 是否使用激活函数
        """
        super().__init__()
        self.in_channels = c1
        self.out_channels = c2
        
        # 计算填充
        if p is None:
            p = k // 2
        
        # 正极性分支（红色通道）：处理像素变亮（正梯度）
        self.pos_branch = nn.Sequential(
            nn.Conv2d(1, c2 // 2, kernel_size=k, stride=s, padding=p, groups=g, dilation=d, bias=False),
            nn.BatchNorm2d(c2 // 2),
            nn.SiLU() if act else nn.Identity()
        )
        
        # 负极性分支（绿色通道）：处理像素变暗（负梯度）
        self.neg_branch = nn.Sequential(
            nn.Conv2d(1, c2 // 2, kernel_size=k, stride=s, padding=p, groups=g, dilation=d, bias=False),
            nn.BatchNorm2d(c2 // 2),
            nn.SiLU() if act else nn.Identity()
        )
        
        # 极性交互模块
        self.interaction = PolarityInteractionModule(c2 // 2, reduction=4)
        
        # 融合卷积：将交互后的特征融合到目标通道数
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(c2 // 2, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU() if act else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x (torch.Tensor): 输入图像 [B, C, H, W]
                如果是RGB图像，C=3，需要提取R和G通道
                如果是灰度图像，需要假设第一个通道是R，第二个是G
        
        Returns:
            (torch.Tensor): 融合后的特征 [B, c2, H, W]
        """
        B, C, H, W = x.shape
        
        # 分离正负极性通道
        if C >= 2:
            # RGB图像：R通道为正极性，G通道为负极性
            pos_channel = x[:, 0:1, :, :]  # 红色通道（正极性）
            neg_channel = x[:, 1:2, :, :]  # 绿色通道（负极性）
        else:
            # 单通道图像：假设输入已经是分离的
            # 这种情况下，需要将输入复制为两个通道
            pos_channel = x
            neg_channel = x
        
        # 正极性分支处理
        pos_feat = self.pos_branch(pos_channel)  # [B, c2//2, H, W]
        
        # 负极性分支处理
        neg_feat = self.neg_branch(neg_channel)  # [B, c2//2, H, W]
        
        # 极性交互
        interaction_feat = self.interaction(pos_feat, neg_feat)  # [B, c2//2, H, W]
        
        # 融合到目标通道数
        output = self.fusion_conv(interaction_feat)  # [B, c2, H, W]
        
        return output
    
    def extra_repr(self) -> str:
        """返回额外表示字符串"""
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}'


def PolarityFusion(c1: int, c2: int, k: int = 3, s: int = 1, p: int = None, 
                   g: int = 1, d: int = 1, act: bool = True, **kwargs) -> DualStreamPolarityFusion:
    """工厂函数：创建双流极性融合模块
    
    Args:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        k (int): 卷积核大小
        s (int): 步长
        p (int, optional): 填充
        g (int): 分组数
        d (int): 膨胀率
        act (bool): 激活函数
        **kwargs: 其他参数
        
    Returns:
        (DualStreamPolarityFusion): 双流极性融合模块实例
    """
    return DualStreamPolarityFusion(c1, c2, k, s, p, g, d, act)
