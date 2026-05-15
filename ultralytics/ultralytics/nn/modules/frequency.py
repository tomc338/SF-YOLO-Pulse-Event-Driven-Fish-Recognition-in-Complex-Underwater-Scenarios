# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Global-Local Frequency YOLO Module

频域与空域双路径学习模块，用于处理ODG图像的高频稀疏特性。
通过FFT/IFFT在频域进行全局滤波，弥补局部卷积无法连接断裂边缘的缺陷。

Reference:
    Rao, Y., et al. "Global Filter Networks for Image Classification." (NeurIPS 2021)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.fft

__all__ = ("GlobalFrequencyFilter", "FrequencyYOLOBlock", "FFTBlock")


class GlobalFrequencyFilter(nn.Module):
    """全局频域滤波器
    
    在频域通过可学习的权重矩阵进行滤波，过滤ODG重建时产生的孤立噪点。
    
    Attributes:
        channels (int): 输入通道数
        use_complex (bool): 是否使用复数权重
    """
    
    def __init__(self, channels: int, use_complex: bool = True):
        """初始化全局频域滤波器
        
        Args:
            channels (int): 输入通道数
            use_complex (bool): 是否使用复数权重（True时更强大但参数更多）
        """
        super().__init__()
        self.channels = channels
        self.use_complex = use_complex
        
        if use_complex:
            # 复数权重：分别学习实部和虚部
            self.weight_real = nn.Parameter(torch.ones(1, channels, 1, 1))
            self.weight_imag = nn.Parameter(torch.zeros(1, channels, 1, 1))
        else:
            # 实数权重：只学习幅度
            self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        
        # 初始化权重
        if use_complex:
            nn.init.xavier_uniform_(self.weight_real)
            nn.init.xavier_uniform_(self.weight_imag)
        else:
            nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x_freq: torch.Tensor) -> torch.Tensor:
        """在频域应用滤波器
        
        Args:
            x_freq (torch.Tensor): 频域特征 [B, C, H, W] (复数)
            
        Returns:
            (torch.Tensor): 滤波后的频域特征 [B, C, H, W] (复数)
        """
        # 保存原始数据类型（用于AMP兼容性）
        original_dtype = x_freq.dtype
        
        # 确保所有操作在FP32精度下进行（AMP兼容性：避免ComplexHalf错误）
        x_freq_fp32 = x_freq.float() if x_freq.dtype != torch.complex64 else x_freq
        
        if self.use_complex:
            # 复数权重滤波
            # 确保权重为FP32类型（AMP可能将参数转换为FP16）
            weight_real_fp32 = self.weight_real.float()
            weight_imag_fp32 = self.weight_imag.float()
            weight = torch.complex(weight_real_fp32, weight_imag_fp32)
            filtered = x_freq_fp32 * weight
        else:
            # 实数权重滤波（只影响幅度）
            weight_fp32 = self.weight.float()
            filtered = x_freq_fp32 * weight_fp32
        
        # 转换回原始数据类型
        if original_dtype != torch.complex64:
            filtered = filtered.to(original_dtype)
        
        return filtered


class FrequencyYOLOBlock(nn.Module):
    """频域YOLO块
    
    将特征图转换到频域，进行全局滤波，然后转回空域。
    实现频域与空域双路径学习。
    
    Attributes:
        channels (int): 输入/输出通道数
        use_complex (bool): 是否使用复数权重
    """
    
    def __init__(self, channels: int, use_complex: bool = True):
        """初始化频域YOLO块
        
        Args:
            channels (int): 输入/输出通道数
            use_complex (bool): 是否使用复数权重
        """
        super().__init__()
        self.channels = channels
        
        # 频域滤波器
        self.freq_filter = GlobalFrequencyFilter(channels, use_complex=use_complex)
        
        # 空域卷积（用于残差连接和特征融合）
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x (torch.Tensor): 输入特征 [B, C, H, W]
            
        Returns:
            (torch.Tensor): 输出特征 [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 保存原始数据类型（用于AMP兼容性）
        original_dtype = x.dtype
        
        # 1. 空域路径（标准卷积）
        spatial_feat = self.spatial_conv(x)
        
        # 2. 频域路径
        # FFT操作需要FP32精度（AMP兼容性：避免ComplexHalf错误）
        # 将输入转换为FP32进行FFT计算
        x_fp32 = x.float()
        
        # FFT: 空域 -> 频域
        x_freq = torch.fft.rfft2(x_fp32, norm='ortho')  # [B, C, H, W//2+1] (复数)
        
        # 频域滤波
        x_freq_filtered = self.freq_filter(x_freq)
        
        # IFFT: 频域 -> 空域
        x_freq_spatial = torch.fft.irfft2(x_freq_filtered, s=(H, W), norm='ortho')  # [B, C, H, W]
        
        # 转换回原始数据类型
        if original_dtype != torch.float32:
            x_freq_spatial = x_freq_spatial.to(original_dtype)
        
        # 3. 双路径融合
        combined = torch.cat([spatial_feat, x_freq_spatial], dim=1)  # [B, 2C, H, W]
        output = self.fusion(combined)  # [B, C, H, W]
        
        # 4. 残差连接
        output = output + x
        
        return output


def FFTBlock(c1: int, c2: int = None, use_complex: bool = True, **kwargs) -> FrequencyYOLOBlock:
    """工厂函数：创建频域YOLO块
    
    Args:
        c1 (int): 输入通道数
        c2 (int, optional): 输出通道数（默认等于c1）
        use_complex (bool): 是否使用复数权重
        **kwargs: 其他参数（忽略）
        
    Returns:
        (FrequencyYOLOBlock): 频域YOLO块实例
    """
    if c2 is None:
        c2 = c1
    if c2 != c1:
        # 如果输出通道数不同，需要先进行通道调整
        return nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            FrequencyYOLOBlock(c2, use_complex=use_complex)
        )
    return FrequencyYOLOBlock(c1, use_complex=use_complex)
