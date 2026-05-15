# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Haar Wavelet Downsampling Module for YOLO
使用2D离散小波变换(DWT)进行下采样，保留高频边缘信息
适用于ES-ImageNet等稀疏边缘图像数据集
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDownsample(nn.Module):
    """
    Haar小波下采样模块
    
    使用2D离散小波变换将输入分解为4个频率子带(LL, LH, HL, HH)，
    并在通道维度拼接，实现信息保留的下采样。
    
    输入: (B, C, H, W)
    输出: (B, C*4, H/2, W/2)
    
    Args:
        in_channels: 输入通道数（用于验证，实际会自动处理）
    """
    
    def __init__(self, in_channels: int = None):
        super().__init__()
        self.in_channels = in_channels
        
        # Haar小波分解的4个卷积核
        # LL (Low-Low): 低频近似
        # LH (Low-High): 水平高频
        # HL (High-Low): 垂直高频  
        # HH (High-High): 对角高频
        
        # 定义Haar小波核 (2x2)
        # 注意：PyTorch卷积核形状为 (out_channels, in_channels, H, W)
        # 这里我们为每个通道创建4个输出通道
        
        # LL核: [1, 1; 1, 1] / 4
        self.register_buffer('ll_kernel', torch.tensor([
            [[[1., 1.], [1., 1.]]]
        ]) / 4.0)
        
        # LH核: [1, -1; 1, -1] / 4
        self.register_buffer('lh_kernel', torch.tensor([
            [[[1., -1.], [1., -1.]]]
        ]) / 4.0)
        
        # HL核: [1, 1; -1, -1] / 4
        self.register_buffer('hl_kernel', torch.tensor([
            [[[1., 1.], [-1., -1.]]]
        ]) / 4.0)
        
        # HH核: [1, -1; -1, 1] / 4
        self.register_buffer('hh_kernel', torch.tensor([
            [[[1., -1.], [-1., 1.]]]
        ]) / 4.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (B, C, H, W)
            
        Returns:
            输出张量 (B, C*4, H/2, W/2)
        """
        B, C, H, W = x.shape
        
        # 处理奇数尺寸：如果H或W是奇数，进行padding
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        
        if pad_h > 0 or pad_w > 0:
            # 使用反射填充，保持边缘信息
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            H, W = x.shape[2], x.shape[3]
        
        # 为每个输入通道创建4个输出通道
        # 使用分组卷积的方式，对每个通道独立应用Haar小波
        
        # 方法1: 使用unfold + 矩阵乘法（更高效）
        # 将输入reshape为 (B*C, 1, H, W) 以便应用卷积
        x_reshaped = x.view(B * C, 1, H, W)
        
        # 对每个通道应用4个小波核
        # 使用conv2d，stride=2进行下采样
        ll = F.conv2d(x_reshaped, self.ll_kernel, stride=2, padding=0)  # (B*C, 1, H/2, W/2)
        lh = F.conv2d(x_reshaped, self.lh_kernel, stride=2, padding=0)  # (B*C, 1, H/2, W/2)
        hl = F.conv2d(x_reshaped, self.hl_kernel, stride=2, padding=0)  # (B*C, 1, H/2, W/2)
        hh = F.conv2d(x_reshaped, self.hh_kernel, stride=2, padding=0)  # (B*C, 1, H/2, W/2)
        
        # 拼接4个子带: (B*C, 4, H/2, W/2)
        subbands = torch.cat([ll, lh, hl, hh], dim=1)  # (B*C, 4, H/2, W/2)
        
        # 重塑回 (B, C*4, H/2, W/2)
        output = subbands.view(B, C * 4, H // 2, W // 2)
        
        return output
    
    def extra_repr(self) -> str:
        return f'in_channels={self.in_channels}'


# 为了兼容ultralytics的parse_model，提供一个便捷函数
def HaarDownsample2d(in_channels: int, out_channels: int = None, **kwargs):
    """
    便捷函数，用于YAML配置
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数（自动为 in_channels * 4，用于验证）
        **kwargs: 其他参数（忽略）
    
    Returns:
        HaarDownsample模块
    """
    if out_channels is not None and out_channels != in_channels * 4:
        # 如果指定了out_channels，验证它是否为in_channels的4倍
        if out_channels != in_channels * 4:
            raise ValueError(
                f"HaarDownsample输出通道数必须是输入通道数的4倍。"
                f"输入: {in_channels}, 期望输出: {in_channels * 4}, 实际输出: {out_channels}"
            )
    return HaarDownsample(in_channels=in_channels)
