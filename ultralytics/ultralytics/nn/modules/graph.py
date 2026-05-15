# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Graph-Augmented YOLO Module

基于图表示的几何推理模块，用于处理ODG图像的稀疏边缘点。
将边缘点视为图节点，利用图卷积进行几何关系推理。

Reference:
    Li, G., et al. "DeepGCNs: Can GCNs Go as Deep as CNNs?" (ICCV 2019)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("GraphReasoningModule", "GraphYOLOBlock", "GCNBlock")


class GraphReasoningModule(nn.Module):
    """图推理模块
    
    将特征图中的关键点提取为图节点，通过图卷积进行信息交换。
    
    Attributes:
        channels (int): 输入通道数
        num_nodes (int): 图节点数量
        num_layers (int): GCN层数
    """
    
    def __init__(self, channels: int, num_nodes: int = 64, num_layers: int = 2):
        """初始化图推理模块
        
        Args:
            channels (int): 输入通道数
            num_nodes (int): 图节点数量（采样点的数量）
            num_layers (int): GCN层数
        """
        super().__init__()
        self.channels = channels
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        
        # 节点采样：从特征图中采样关键点作为图节点
        self.node_sampler = nn.AdaptiveAvgPool2d((int(num_nodes ** 0.5), int(num_nodes ** 0.5)))
        
        # GCN层：图卷积网络
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gcn_layers.append(
                nn.Sequential(
                    nn.Linear(channels, channels, bias=False),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # 邻接矩阵：可学习的图结构
        self.adjacency = nn.Parameter(torch.eye(num_nodes))
        
        # 特征投影：将图特征投影回原始空间
        self.feature_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
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
        
        # 1. 节点采样：从特征图中采样关键点
        # 使用自适应池化采样固定数量的节点
        nodes = self.node_sampler(x)  # [B, C, sqrt(N), sqrt(N)]
        N = self.num_nodes
        nodes = nodes.view(B, C, N).permute(0, 2, 1)  # [B, N, C]
        
        # 2. 图卷积：在节点间进行信息交换
        # 构建邻接矩阵（添加自连接）
        # 确保邻接矩阵与输入数据类型一致（AMP兼容性）
        adj = self.adjacency.float() + torch.eye(N, device=x.device, dtype=torch.float32)
        # 归一化邻接矩阵
        adj = F.normalize(adj, p=1, dim=1)
        
        # 确保nodes也是FP32（AMP兼容性）
        nodes_fp32 = nodes.float()
        
        # GCN前向传播：X' = AXW
        # 确保整个GCN计算都在FP32下进行（AMP兼容性）
        graph_feat = nodes_fp32.float() if nodes_fp32.dtype != torch.float32 else nodes_fp32
        for gcn_layer in self.gcn_layers:
            # 图卷积：AX
            # 确保adj和graph_feat都是FP32（AMP兼容性）
            adj_expanded = adj.unsqueeze(0).expand(B, -1, -1)  # [B, N, N]
            graph_feat = torch.bmm(adj_expanded, graph_feat.float())  # [B, N, C] (确保FP32)
            # 线性变换：W
            # 确保输入和权重都是FP32（AMP兼容性）
            graph_feat = graph_feat.reshape(-1, C)
            # 将输入转换为FP32，然后调用GCN层
            # GCN层的权重在AMP下可能是FP16，所以我们需要手动处理
            graph_feat_fp32 = graph_feat.float() if graph_feat.dtype != torch.float32 else graph_feat
            
            # 手动执行Linear操作以确保数据类型一致
            linear_layer = gcn_layer[0]  # nn.Linear
            bn_layer = gcn_layer[1]  # nn.BatchNorm1d
            relu_layer = gcn_layer[2]  # nn.ReLU
            
            # Linear: 确保输入和权重都是FP32
            weight_fp32 = linear_layer.weight.float() if linear_layer.weight.dtype != torch.float32 else linear_layer.weight
            graph_feat_fp32 = F.linear(graph_feat_fp32, weight_fp32, None)
            
            # BatchNorm和ReLU: 确保BatchNorm的所有参数都是FP32
            # 手动执行BatchNorm以确保数据类型一致
            # 强制将所有参数转换为FP32
            running_mean = bn_layer.running_mean.float()
            running_var = bn_layer.running_var.float()
            weight_bn = bn_layer.weight.float() if bn_layer.weight is not None else None
            bias_bn = bn_layer.bias.float() if bn_layer.bias is not None else None
            
            # 确保输入也是FP32
            graph_feat_fp32_input = graph_feat_fp32.float() if graph_feat_fp32.dtype != torch.float32 else graph_feat_fp32
            
            graph_feat_fp32 = F.batch_norm(
                graph_feat_fp32_input,
                running_mean,
                running_var,
                weight_bn,
                bias_bn,
                bn_layer.training,
                bn_layer.momentum,
                bn_layer.eps
            )
            
            graph_feat_fp32 = relu_layer(graph_feat_fp32)
            
            graph_feat = graph_feat_fp32.reshape(B, N, C)
        
        # 3. 特征投影：将图特征投影回原始空间
        graph_feat = graph_feat.permute(0, 2, 1).view(B, C, int(N ** 0.5), int(N ** 0.5))  # [B, C, sqrt(N), sqrt(N)]
        graph_feat = F.interpolate(graph_feat, size=(H, W), mode='bilinear', align_corners=False)  # [B, C, H, W]
        
        # 特征投影：确保输入和权重都是FP32（AMP兼容性）
        # 手动执行Conv2d, BatchNorm2d, ReLU以确保数据类型一致
        graph_feat_fp32 = graph_feat.float() if graph_feat.dtype != torch.float32 else graph_feat
        
        # Conv2d
        conv_layer = self.feature_proj[0]  # nn.Conv2d
        weight_conv = conv_layer.weight.float() if conv_layer.weight.dtype != torch.float32 else conv_layer.weight
        bias_conv = conv_layer.bias.float() if conv_layer.bias is not None and conv_layer.bias.dtype != torch.float32 else (conv_layer.bias if conv_layer.bias is None else conv_layer.bias)
        graph_feat_fp32 = F.conv2d(
            graph_feat_fp32,
            weight_conv,
            bias_conv,
            conv_layer.stride,
            conv_layer.padding,
            conv_layer.dilation,
            conv_layer.groups
        )
        
        # BatchNorm2d
        bn_layer = self.feature_proj[1]  # nn.BatchNorm2d
        running_mean = bn_layer.running_mean.float()
        running_var = bn_layer.running_var.float()
        weight_bn = bn_layer.weight.float() if bn_layer.weight is not None else None
        bias_bn = bn_layer.bias.float() if bn_layer.bias is not None else None
        graph_feat_fp32 = F.batch_norm(
            graph_feat_fp32,
            running_mean,
            running_var,
            weight_bn,
            bias_bn,
            bn_layer.training,
            bn_layer.momentum,
            bn_layer.eps
        )
        
        # ReLU
        relu_layer = self.feature_proj[2]  # nn.ReLU
        graph_feat_fp32 = relu_layer(graph_feat_fp32)
        
        # 转换回原始数据类型
        if original_dtype != torch.float32:
            graph_feat_fp32 = graph_feat_fp32.to(original_dtype)
        
        # 4. 与原始特征融合
        output = x + graph_feat_fp32
        
        return output


class GraphYOLOBlock(nn.Module):
    """图增强YOLO块
    
    结合标准卷积和图推理模块。
    """
    
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int = None, 
                 g: int = 1, d: int = 1, act: bool = True, num_nodes: int = 64):
        """初始化图增强YOLO块
        
        Args:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            k (int): 卷积核大小
            s (int): 步长
            p (int, optional): 填充
            g (int): 分组数
            d (int): 膨胀率
            act (bool): 是否使用激活函数
            num_nodes (int): 图节点数量
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
        
        # 图推理模块（仅在通道数匹配且不降采样时使用）
        if c1 == c2 and s == 1:
            self.graph_reasoning = GraphReasoningModule(c2, num_nodes=num_nodes)
        else:
            self.graph_reasoning = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.conv(x)
        x = self.graph_reasoning(x)
        return x


def GCNBlock(c1: int, c2: int, k: int = 3, s: int = 1, p: int = None, 
             g: int = 1, d: int = 1, act: bool = True, num_nodes: int = 64, **kwargs) -> GraphYOLOBlock:
    """工厂函数：创建图增强YOLO块
    
    Args:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        k (int): 卷积核大小
        s (int): 步长
        p (int, optional): 填充
        g (int): 分组数
        d (int): 膨胀率
        act (bool): 是否使用激活函数
        num_nodes (int): 图节点数量
        **kwargs: 其他参数（忽略）
    """
    return GraphYOLOBlock(c1, c2, k, s, p, g, d, act, num_nodes=num_nodes)
