# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Lateral Inhibition Module

仿生"侧向抑制"机制模块，用于锐化边缘、消除背景干扰。
借鉴生物视觉系统的侧向抑制原理，增强强边缘点，抑制孤立的弱噪声点。

Reference:
    Wang, L., et al. "Biologically Inspired Lateral Inhibition Network for Edge Detection."
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("LateralInhibitionModule", "LIM", "LateralInhibitionBlock")


class LateralInhibitionModule(nn.Module):
    """侧向抑制模块
    
    通过竞争机制增强强边缘点，抑制孤立的弱噪声点。
    类似Non-Local但权重矩阵经过特殊处理，实现侧向抑制效果。
    
    Attributes:
        channels (int): 输入通道数
        reduction (int): 通道缩减比例
        kernel_size (int): 局部抑制核大小
    """
    
    def __init__(self, channels: int, reduction: int = 4, kernel_size: int = 3):
        """初始化侧向抑制模块
        
        Args:
            channels (int): 输入通道数
            reduction (int): 通道缩减比例
            kernel_size (int): 局部抑制核大小
        """
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        
        # 通道注意力：识别强特征
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # 空间抑制：抑制周围弱特征
        # 使用可分离卷积实现高效的局部抑制
        self.spatial_inhibition = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2, 
                     groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()  # 输出抑制权重（0-1之间）
        )
        
        # 全局竞争：Non-Local风格的全局关系
        self.global_competition = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x (torch.Tensor): 输入特征 [B, C, H, W]
            
        Returns:
            (torch.Tensor): 输出特征 [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 1. 通道注意力：识别强特征通道
        channel_att = self.channel_attention(x)  # [B, C, 1, 1]
        x_channel = x * channel_att
        
        # 2. 空间抑制：抑制周围弱特征
        # 计算局部抑制权重（强特征抑制弱特征）
        inhibition_weight = self.spatial_inhibition(x_channel)  # [B, C, H, W]
        # 反转：强特征得到高权重，弱特征得到低权重
        enhancement_weight = 1.0 - inhibition_weight
        x_spatial = x_channel * enhancement_weight
        
        # 3. 全局竞争：全局尺度上的特征竞争
        # 计算全局平均池化
        x_global = F.adaptive_avg_pool2d(x_spatial, 1)  # [B, C, 1, 1]
        global_comp = self.global_competition(x_global)  # [B, C, 1, 1]
        x_global_enhanced = x_spatial * global_comp
        
        # 4. 输出投影
        output = self.output_proj(x_global_enhanced)
        
        # 5. 残差连接
        output = output + x
        
        return output


class LateralInhibitionBlock(nn.Module):
    """侧向抑制块（带卷积的完整模块）
    
    结合标准卷积和侧向抑制模块。
    """
    
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int = None, 
                 g: int = 1, d: int = 1, act: bool = True):
        """初始化侧向抑制块
        
        Args:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            k (int): 卷积核大小
            s (int): 步长
            p (int, optional): 填充
            g (int): 分组数
            d (int): 膨胀率
            act (bool): 是否使用激活函数
        """
        super().__init__()
        
        # 计算填充
        if p is None:
            p = k // 2
        
        # 标准卷积
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU() if act else nn.Identity()
        )
        
        # 侧向抑制模块（仅在通道数匹配时使用）
        if c1 == c2 and s == 1:
            self.lim = LateralInhibitionModule(c2)
        else:
            self.lim = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.conv(x)
        x = self.lim(x)
        return x


def LIM(c1: int, c2: int, k: int = 3, s: int = 1, p: int = None, 
        g: int = 1, d: int = 1, act: bool = True, **kwargs) -> LateralInhibitionBlock:
    """工厂函数：创建侧向抑制块"""
    return LateralInhibitionBlock(c1, c2, k, s, p, g, d, act)

# 别名
LateralInhibition = LateralInhibitionBlock
