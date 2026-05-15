# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
SEAM (Stacked Element-wise Attention Module) Module

SEAM module is designed for contextual information extraction, particularly effective
for small object detection and low-resolution images. It uses multi-scale depthwise
separable convolutions to simulate patch partitioning and cross-scale feature merging.

Reference:
    Lyu, C., et al. "SEAM: Stacked Element-wise Attention Module for Contextual Information"
    CVPR 2020 Workshop / NTIRE 2020

Key Components:
    1. Multi-scale Patch Partitioning: Uses different kernel sizes (3x3, 5x5, 7x7) 
       of depthwise separable convolutions to capture multi-scale features
    2. CSMM (Cross-Scale Merging Module): Merges features from different scales
    3. Residual Connection: Preserves original information and prevents gradient vanishing
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv

__all__ = ("SEAM", "SEAMBlock", "CSMM", "MultiScaleDWSepConv")


class MultiScaleDWSepConv(nn.Module):
    """Multi-scale Depthwise Separable Convolution for patch-like feature extraction.
    
    This module uses different kernel sizes to simulate multi-scale patch partitioning,
    capturing features at different receptive fields.
    
    Attributes:
        branches (nn.ModuleList): List of depthwise separable convolution branches.
    """
    
    def __init__(self, c1: int, c2: int, kernel_sizes: list = [3, 5, 7], act: bool = True):
        """Initialize multi-scale depthwise separable convolution.
        
        Args:
            c1 (int): Input channels.
            c2 (int): Output channels per branch.
            kernel_sizes (list): List of kernel sizes for different scales. Default: [3, 5, 7].
            act (bool): Whether to use activation function.
        """
        super().__init__()
        self.num_branches = len(kernel_sizes)
        
        # Create depthwise separable convolution branches
        # Each branch: 1x1 pointwise conv -> depthwise conv -> 1x1 pointwise conv
        # 确保所有分支使用相同的padding策略，保持输出尺寸一致
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            # DWConv会自动计算padding，确保输出尺寸与输入一致（stride=1时）
            branch = nn.Sequential(
                Conv(c1, c2, k=1, s=1, act=act),  # Pointwise conv 1
                DWConv(c2, c2, k=k, s=1, act=act),  # Depthwise conv (自动padding)
                Conv(c2, c2, k=1, s=1, act=act)  # Pointwise conv 2
            )
            self.branches.append(branch)
    
    def forward(self, x: torch.Tensor) -> list:
        """Forward pass through multi-scale branches.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            list: List of feature maps from different scales, each of shape (B, c2, H, W).
        """
        outputs = []
        for branch in self.branches:
            out = branch(x)
            outputs.append(out)
        return outputs


class CSMM(nn.Module):
    """Cross-Scale Merging Module (CSMM).
    
    This module merges features from different scales to enhance cross-scale
    feature relationships and capture contextual information.
    
    Attributes:
        merge_conv (Conv): Convolution layer for merging multi-scale features.
        attention (nn.Module): Element-wise attention mechanism.
    """
    
    def __init__(self, c: int, num_scales: int = 3, reduction: int = 4, act: bool = True):
        """Initialize Cross-Scale Merging Module.
        
        Args:
            c (int): Number of channels per scale.
            num_scales (int): Number of input scales. Default: 3.
            reduction (int): Reduction ratio for attention. Default: 4.
            act (bool): Whether to use activation function.
        """
        super().__init__()
        self.num_scales = num_scales
        self.c = c
        
        # Merge multi-scale features
        self.merge_conv = Conv(c * num_scales, c, k=1, s=1, act=act)
        
        # Element-wise attention mechanism
        # Global average pooling -> FC layers -> Sigmoid
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(c, c // reduction, k=1, s=1, act=True),
            Conv(c // reduction, c, k=1, s=1, act=False),
            nn.Sigmoid()
        )
    
    def forward(self, features: list) -> torch.Tensor:
        """Forward pass through CSMM.
        
        Args:
            features (list): List of feature maps from different scales, 
                          each of shape (B, c, H, W).
            
        Returns:
            torch.Tensor: Merged and attention-weighted features of shape (B, c, H, W).
        """
        if len(features) == 0:
            raise ValueError("CSMM requires at least one feature map")
        
        # Concatenate multi-scale features
        x = torch.cat(features, dim=1)  # (B, c*num_scales, H, W)
        
        # Merge features
        x = self.merge_conv(x)  # (B, c, H, W)
        
        # Apply element-wise attention
        attn = self.attention(x)  # (B, c, 1, 1)
        x = x * attn  # Element-wise multiplication
        
        return x


class SEAM(nn.Module):
    """SEAM (Stacked Element-wise Attention Module).
    
    Complete SEAM module combining multi-scale depthwise separable convolutions,
    cross-scale merging, and residual connection.
    
    Attributes:
        multi_scale_conv (MultiScaleDWSepConv): Multi-scale feature extraction.
        csmm (CSMM): Cross-scale merging module.
        residual_conv (Conv): Optional residual connection projection.
    """
    
    def __init__(self, c1: int, c2: int = None, kernel_sizes: list = [3, 5, 7], 
                 reduction: int = 4, act: bool = True, use_residual: bool = True):
        """Initialize SEAM module.
        
        Args:
            c1 (int): Input channels.
            c2 (int): Output channels. If None, set to c1. Default: None.
            kernel_sizes (list): Kernel sizes for multi-scale branches. Default: [3, 5, 7].
            reduction (int): Reduction ratio for attention. Default: 4.
            act (bool): Whether to use activation function. Default: True.
            use_residual (bool): Whether to use residual connection. Default: True.
        """
        super().__init__()
        if c2 is None:
            c2 = c1
        
        self.c1 = c1
        self.c2 = c2
        self.use_residual = use_residual
        
        # Multi-scale depthwise separable convolution
        self.multi_scale_conv = MultiScaleDWSepConv(c1, c2, kernel_sizes, act)
        
        # Cross-scale merging module
        self.csmm = CSMM(c2, num_scales=len(kernel_sizes), reduction=reduction, act=act)
        
        # Residual connection projection (if channel dimensions differ)
        if use_residual:
            if c1 != c2:
                self.residual_conv = Conv(c1, c2, k=1, s=1, act=False)
            else:
                self.residual_conv = nn.Identity()
        else:
            self.residual_conv = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SEAM module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, c1, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, c2, H, W).
        """
        B, C, H, W = x.shape
        
        # Store input for residual connection
        identity = x if self.use_residual else None
        
        # Multi-scale feature extraction
        multi_scale_features = self.multi_scale_conv(x)
        
        # Cross-scale merging with attention
        out = self.csmm(multi_scale_features)
        
        # Residual connection
        if self.use_residual:
            if self.residual_conv is not None:
                identity = self.residual_conv(identity)
            out = out + identity
        
        return out


def SEAMBlock(c1: int, c2: int = None, kernel_sizes: list = [3, 5, 7], 
              reduction: int = 4, act: bool = True, **kwargs) -> SEAM:
    """Convenience function to create SEAM block.
    
    Args:
        c1 (int): Input channels.
        c2 (int): Output channels. If None, set to c1.
        kernel_sizes (list): Kernel sizes for multi-scale branches.
        reduction (int): Reduction ratio for attention.
        act (bool): Whether to use activation function.
        **kwargs: Additional arguments (ignored).
        
    Returns:
        SEAM: SEAM module instance.
    """
    return SEAM(c1, c2, kernel_sizes, reduction, act)
