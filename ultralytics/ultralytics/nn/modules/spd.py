# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
SPD-Conv (Space-to-Depth Convolution) Module

SPD-Conv replaces strided convolutions and pooling operations to prevent information loss
in small object detection. It uses space-to-depth transformation to preserve all spatial information.

Reference:
    Sunkara, R., et al. "No More Strided Convolutions or Pooling: A New CNN Building Block 
    for Low-Resolution Images and Small Objects." WACV 2022/2023
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("SPDConv", "SPDConv2d")


class SPDConv(nn.Module):
    """SPD-Conv (Space-to-Depth Convolution) module.
    
    This module replaces standard strided convolutions by:
    1. Splitting the input into non-overlapping patches
    2. Rearranging spatial dimensions into channel dimensions (space-to-depth)
    3. Applying a standard convolution without stride
    
    This preserves all spatial information, making it ideal for small object detection
    and edge-based features (like ODG images).
    
    Attributes:
        conv (nn.Conv2d): Standard convolution layer (no stride).
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function.
        scale_factor (int): Downsampling factor (typically 2).
    """
    
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2, p: int = None, g: int = 1, 
                 d: int = 1, act: bool = True, scale_factor: int = 2):
        """Initialize SPD-Conv module.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size (default: 3).
            s (int): Stride (should be 2 for downsampling, but SPD-Conv uses scale_factor instead).
            p (int, optional): Padding.
            g (int): Groups for convolution.
            d (int): Dilation.
            act (bool): Whether to apply activation.
            scale_factor (int): Downsampling factor (default: 2).
        """
        super().__init__()
        self.scale_factor = scale_factor
        
        # Calculate padding for 'same' output
        if p is None:
            p = k // 2
        
        # Standard convolution without stride (SPD handles downsampling)
        self.conv = nn.Conv2d(
            c1 * (scale_factor ** 2),  # Input channels multiplied by scale_factor^2
            c2, 
            k, 
            stride=1,  # No stride - downsampling is done by SPD
            padding=p, 
            groups=g, 
            dilation=d, 
            bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SPD-Conv.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            (torch.Tensor): Output tensor of shape (B, C2, H//scale_factor, W//scale_factor).
        """
        B, C, H, W = x.shape
        
        # Handle odd dimensions by padding
        if H % self.scale_factor != 0 or W % self.scale_factor != 0:
            pad_h = (self.scale_factor - H % self.scale_factor) % self.scale_factor
            pad_w = (self.scale_factor - W % self.scale_factor) % self.scale_factor
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w
        
        # Space-to-Depth transformation
        # Split into non-overlapping patches and rearrange to channels
        x = x.view(B, C, H // self.scale_factor, self.scale_factor, W // self.scale_factor, self.scale_factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()  # (B, C, scale, scale, H//scale, W//scale)
        x = x.view(B, C * (self.scale_factor ** 2), H // self.scale_factor, W // self.scale_factor)
        
        # Apply standard convolution (no stride)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        
        return x
    
    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f'in_channels={self.conv.in_channels // (self.scale_factor ** 2)}, ' \
               f'out_channels={self.conv.out_channels}, ' \
               f'kernel_size={self.conv.kernel_size}, ' \
               f'scale_factor={self.scale_factor}'


def SPDConv2d(c1: int, c2: int, k: int = 3, s: int = 2, p: int = None, g: int = 1, 
              d: int = 1, act: bool = True, **kwargs) -> SPDConv:
    """Factory function for SPD-Conv module.
    
    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        k (int): Kernel size.
        s (int): Stride (used to determine scale_factor).
        p (int, optional): Padding.
        g (int): Groups.
        d (int): Dilation.
        act (bool): Activation.
        **kwargs: Additional keyword arguments.
        
    Returns:
        (SPDConv): SPD-Conv module instance.
    """
    scale_factor = s if s > 1 else 2  # Default to 2 if stride is 1
    return SPDConv(c1, c2, k, s, p, g, d, act, scale_factor=scale_factor)
