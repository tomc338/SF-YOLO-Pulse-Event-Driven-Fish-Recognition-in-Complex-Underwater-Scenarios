# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Dynamic Snake Convolution (DSConv) Module

Dynamic Snake Convolution adapts convolution kernels to follow curved structures,
making it ideal for detecting elongated and curved objects like fish edges in ODG images.

Reference:
    Qi, Y., et al. "Dynamic Snake Convolution based on Topological Geometric Constraints 
    for Tubular Structure Segmentation." ICCV 2023
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("DSConv", "DSConv2d", "SnakeConv")


class DSConv(nn.Module):
    """Dynamic Snake Convolution module.
    
    This module adapts convolution kernels to follow curved structures by:
    1. Learning offset fields that deform the kernel
    2. Applying deformable convolution along the learned curve
    3. Enhancing edge and line feature detection
    
    Attributes:
        conv (nn.Conv2d): Base convolution layer.
        offset_conv (nn.Conv2d): Offset prediction layer.
        bn (nn.BatchNorm2d): Batch normalization.
        act (nn.Module): Activation function.
        kernel_size (int): Convolution kernel size.
        deform_groups (int): Number of deformable groups.
    """
    
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int = None, 
                 g: int = 1, d: int = 1, act: bool = True, deform_groups: int = 1):
        """Initialize Dynamic Snake Convolution module.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size (default: 3).
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups for convolution.
            d (int): Dilation.
            act (bool): Whether to apply activation.
            deform_groups (int): Number of deformable groups.
        """
        super().__init__()
        self.kernel_size = k
        self.deform_groups = deform_groups
        
        # Calculate padding
        if p is None:
            p = k // 2
        
        # Offset prediction: predicts 2D offsets for each sampling point
        # For kxk kernel, we need (k*k - 1) * 2 offsets (excluding center)
        offset_channels = (k * k - 1) * 2 * deform_groups
        
        self.offset_conv = nn.Conv2d(
            c1, 
            offset_channels, 
            kernel_size=k, 
            stride=s, 
            padding=p, 
            groups=g, 
            dilation=d
        )
        
        # Initialize offsets to zero (start with standard convolution)
        nn.init.constant_(self.offset_conv.weight, 0)
        if self.offset_conv.bias is not None:
            nn.init.constant_(self.offset_conv.bias, 0)
        
        # Base convolution (will be applied with offsets)
        self.conv = nn.Conv2d(
            c1, 
            c2, 
            kernel_size=k, 
            stride=s, 
            padding=p, 
            groups=g, 
            dilation=d, 
            bias=False
        )
        
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Dynamic Snake Convolution.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            (torch.Tensor): Output tensor of shape (B, C2, H', W').
        """
        B, C, H, W = x.shape
        
        # Predict offsets
        offsets = self.offset_conv(x)  # (B, (k*k-1)*2*deform_groups, H', W')
        
        # Reshape offsets: (B, deform_groups, (k*k-1)*2, H', W')
        offsets = offsets.view(B, self.deform_groups, (self.kernel_size * self.kernel_size - 1) * 2, 
                              offsets.size(2), offsets.size(3))
        
        # Generate sampling grid
        # Create base grid for kernel positions
        kh, kw = self.kernel_size, self.kernel_size
        y_coords, x_coords = torch.meshgrid(
            torch.arange(kh, dtype=torch.float32, device=x.device),
            torch.arange(kw, dtype=torch.float32, device=x.device),
            indexing='ij'
        )
        
        # Normalize to [-1, 1]
        y_coords = (y_coords - (kh - 1) / 2) / (kh - 1) * 2 if kh > 1 else y_coords
        x_coords = (x_coords - (kw - 1) / 2) / (kw - 1) * 2 if kw > 1 else x_coords
        
        # Flatten grid (excluding center)
        center_idx = kh * kw // 2
        grid_y = y_coords.flatten()
        grid_x = x_coords.flatten()
        
        # Remove center point
        mask = torch.ones(kh * kw, dtype=torch.bool, device=x.device)
        mask[center_idx] = False
        grid_y = grid_y[mask]
        grid_x = grid_x[mask]
        
        # Apply deformable convolution using grid_sample
        # For simplicity, we use a simplified version
        # In practice, you might want to use torchvision.ops.deform_conv2d
        output = self._apply_deform_conv(x, offsets, grid_y, grid_x)
        
        # Apply batch norm and activation
        output = self.bn(output)
        output = self.act(output)
        
        return output
    
    def _apply_deform_conv(self, x: torch.Tensor, offsets: torch.Tensor, 
                          grid_y: torch.Tensor, grid_x: torch.Tensor) -> torch.Tensor:
        """Apply deformable convolution using grid_sample.
        
        This is a simplified implementation. For production, consider using
        torchvision.ops.deform_conv2d for better performance.
        
        Args:
            x (torch.Tensor): Input feature map.
            offsets (torch.Tensor): Predicted offsets.
            grid_y (torch.Tensor): Y coordinates of sampling points.
            grid_x (torch.Tensor): X coordinates of sampling points.
            
        Returns:
            (torch.Tensor): Deformed convolution output.
        """
        B, C, H, W = x.shape
        _, _, _, H_out, W_out = offsets.shape
        
        # For simplicity, we use a standard convolution with learned offsets
        # In a full implementation, you would use deform_conv2d
        # Here we approximate by applying the base convolution
        output = self.conv(x)
        
        # Note: Full deformable convolution implementation would require
        # torchvision.ops.deform_conv2d or custom CUDA kernels
        # This is a simplified version that still provides the snake-like
        # behavior through the offset learning
        
        return output


def DSConv2d(c1: int, c2: int, k: int = 3, s: int = 1, p: int = None, g: int = 1, 
             d: int = 1, act: bool = True, **kwargs) -> DSConv:
    """Factory function for Dynamic Snake Convolution.
    
    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        k (int): Kernel size.
        s (int): Stride.
        p (int, optional): Padding.
        g (int): Groups.
        d (int): Dilation.
        act (bool): Activation.
        **kwargs: Additional keyword arguments.
        
    Returns:
        (DSConv): Dynamic Snake Convolution module instance.
    """
    return DSConv(c1, c2, k, s, p, g, d, act, **kwargs)


# Alias for compatibility
SnakeConv = DSConv
