# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Structural Connectivity Module for Event-Based Vision

This module explicitly models structural connectivity in simulated event images,
where objects are represented by sparse event points forming contours rather than dense textures.

Key Insight:
In simulated event images, individual points are not semantically meaningful. 
Object identity emerges from connectivity and continuity of high-response event points.
Therefore, the network should first learn whether points belong to the same 
structural contour before bounding box regression.

Architecture:
1. Node Construction: Extract high-activation spatial locations as graph nodes
2. Local Graph Building: KNN or radius-based neighborhood (non-learnable)
3. Connectivity-Aware Message Passing: Lightweight GNN to learn contour membership
4. Structural Feature Re-projection: Re-project node features back to feature map
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

__all__ = ("ConnectivityAwareModule", "ConnectivityBlock")


class ConnectivityAwareModule(nn.Module):
    """
    Connectivity-Aware Module for structural contour formation.
    
    This module treats high-activation spatial locations as graph nodes and learns
    whether nodes belong to the same structural contour through message passing.
    
    Attributes:
        channels (int): Number of input/output channels
        num_nodes (int): Maximum number of nodes to extract
        k_neighbors (int): Number of neighbors for KNN graph construction
        message_passing_layers (int): Number of GNN layers for message passing
    """
    
    def __init__(
        self,
        channels: int,
        num_nodes: int = 128,
        k_neighbors: int = 8,
        message_passing_layers: int = 2,
        activation_threshold: float = 0.1,
    ):
        """
        Initialize Connectivity-Aware Module.
        
        Args:
            channels (int): Number of input/output channels
            num_nodes (int): Maximum number of nodes to extract from high-activation regions
            k_neighbors (int): Number of neighbors for KNN graph construction
            message_passing_layers (int): Number of GNN layers for message passing
            activation_threshold (float): Threshold for extracting high-activation nodes
        """
        super().__init__()
        self.channels = channels
        self.num_nodes = num_nodes
        self.k_neighbors = k_neighbors
        self.message_passing_layers = message_passing_layers
        self.activation_threshold = activation_threshold
        
        # Node feature projection: combines local features with spatial coordinates
        # Input: [B, C, H, W] -> Node features: [B, N, C+2] (C channels + 2 spatial coords)
        self.node_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Message passing layers: learn whether nodes belong to same contour
        # Each layer performs: h_i' = f(h_i, aggregate({h_j : j in N(i)}))
        self.message_passing = nn.ModuleList()
        for _ in range(message_passing_layers):
            # Edge aggregation: aggregate neighbor features
            self.message_passing.append(
                nn.Sequential(
                    nn.Linear(channels + 2, channels, bias=False),  # +2 for spatial coords
                    nn.BatchNorm1d(channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(channels, channels, bias=False),
                    nn.BatchNorm1d(channels),
                )
            )
        
        # Node update: combine original and aggregated features
        self.node_update = nn.Sequential(
            nn.Linear(channels * 2, channels, bias=False),  # original + aggregated
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Feature re-projection: map enhanced node features back to spatial grid
        self.reproject = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def _extract_nodes(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract high-activation spatial locations as graph nodes.
        
        Args:
            x: Input feature map [B, C, H, W]
            mask: Optional binary mask [B, H, W] for node extraction
            
        Returns:
            node_features: Node feature vectors [B, N, C+2] (includes spatial coords)
            node_coords: Normalized spatial coordinates [B, N, 2]
            node_indices: Spatial indices of nodes [B, N, 2] (for re-projection)
        """
        B, C, H, W = x.shape
        
        # Compute activation map: channel-wise max pooling
        activation = x.abs().max(dim=1, keepdim=True)[0]  # [B, 1, H, W]
        
        # Apply threshold to get high-activation regions
        if mask is not None:
            activation = activation * mask.unsqueeze(1)
        
        # Flatten spatial dimensions
        activation_flat = activation.view(B, -1)  # [B, H*W]
        
        # Select top-k high-activation locations as nodes
        # Use adaptive threshold: select top num_nodes locations
        _, top_indices = torch.topk(activation_flat, min(self.num_nodes, H * W), dim=1)  # [B, N]
        
        # Convert flat indices to 2D coordinates
        node_y = top_indices // W  # [B, N]
        node_x = top_indices % W   # [B, N]
        
        # Normalize coordinates to [0, 1]
        node_coords = torch.stack([
            node_x.float() / (W - 1) if W > 1 else node_x.float(),
            node_y.float() / (H - 1) if H > 1 else node_y.float()
        ], dim=2)  # [B, N, 2]
        
        # Extract node features: project input features
        x_proj = self.node_proj(x)  # [B, C, H, W]
        
        # Gather features at node locations
        batch_indices = torch.arange(B, device=x.device).view(B, 1, 1).expand(-1, top_indices.shape[1], 1)
        node_indices_3d = torch.stack([batch_indices.squeeze(-1), node_y, node_x], dim=2)  # [B, N, 3]
        
        # Flatten for indexing
        x_flat = x_proj.view(B, C, -1)  # [B, C, H*W]
        node_features = x_flat.gather(2, top_indices.unsqueeze(1).expand(-1, C, -1))  # [B, C, N]
        node_features = node_features.permute(0, 2, 1)  # [B, N, C]
        
        # Concatenate spatial coordinates to node features
        node_features = torch.cat([node_features, node_coords], dim=2)  # [B, N, C+2]
        
        return node_features, node_coords, torch.stack([node_y, node_x], dim=2)  # [B, N, 2]
    
    def _build_knn_graph(
        self, 
        node_coords: torch.Tensor,
        node_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Build K-Nearest Neighbors graph (non-learnable).
        
        Args:
            node_coords: Normalized spatial coordinates [B, N, 2]
            node_features: Node features [B, N, C+2]
            
        Returns:
            adjacency: Binary adjacency matrix [B, N, N]
        """
        B, N, _ = node_coords.shape
        
        # Compute pairwise distances in feature + spatial space
        # Distance = ||feature_i - feature_j||^2 + alpha * ||coord_i - coord_j||^2
        node_features_expanded_i = node_features.unsqueeze(2)  # [B, N, 1, C+2]
        node_features_expanded_j = node_features.unsqueeze(1)  # [B, 1, N, C+2]
        
        # Feature distance
        feature_dist = torch.sum((node_features_expanded_i - node_features_expanded_j) ** 2, dim=3)  # [B, N, N]
        
        # Spatial distance (weighted)
        coord_dist = torch.sum((node_coords.unsqueeze(2) - node_coords.unsqueeze(1)) ** 2, dim=3)  # [B, N, N]
        
        # Combined distance
        combined_dist = feature_dist + 0.5 * coord_dist  # [B, N, N]
        
        # Find k-nearest neighbors for each node
        _, knn_indices = torch.topk(combined_dist, min(self.k_neighbors + 1, N), dim=2, largest=False)  # [B, N, k+1]
        # +1 because node itself is included
        
        # Build binary adjacency matrix (non-inplace to avoid gradient issues)
        adjacency = torch.zeros(B, N, N, device=node_coords.device, dtype=torch.float32)
        batch_indices = torch.arange(B, device=node_coords.device).view(B, 1, 1)
        node_indices = torch.arange(N, device=node_coords.device).view(1, N, 1)
        
        # Set edges for k-nearest neighbors (use non-inplace scatter)
        adjacency = adjacency.scatter(2, knn_indices, 1.0)
        
        # Remove self-connections (optional, but we keep them for residual)
        # adjacency = adjacency.scatter(2, node_indices.expand(B, N, 1), 0.0)
        
        # Normalize adjacency matrix (row normalization)
        degree = adjacency.sum(dim=2, keepdim=True) + 1e-8
        adjacency = adjacency / degree
        
        return adjacency
    
    def _message_passing(
        self, 
        node_features: torch.Tensor,
        adjacency: torch.Tensor
    ) -> torch.Tensor:
        """
        Connectivity-aware message passing to learn contour membership.
        
        Args:
            node_features: Node features [B, N, C+2]
            adjacency: Adjacency matrix [B, N, N]
            
        Returns:
            enhanced_features: Enhanced node features [B, N, C]
        """
        B, N, _ = node_features.shape
        
        # Store original features (without spatial coords)
        original_features = node_features[:, :, :-2]  # [B, N, C]
        
        # Message passing through GNN layers
        current_features = node_features
        for layer in self.message_passing:
            # Aggregate neighbor messages: A @ X
            aggregated = torch.bmm(adjacency, current_features)  # [B, N, C+2]
            
            # Transform aggregated messages
            aggregated_flat = aggregated.view(-1, self.channels + 2)  # [BN, C+2]
            transformed = layer(aggregated_flat)  # [BN, C]
            transformed = transformed.view(B, N, self.channels)  # [B, N, C]
            
            # Update current features: combine original and aggregated
            current_features_combined = torch.cat([
                original_features,
                transformed
            ], dim=2)  # [B, N, 2C]
            
            # Node update: combine original and aggregated
            current_features_flat = current_features_combined.view(-1, 2 * self.channels)  # [BN, 2C]
            current_features = self.node_update(current_features_flat)  # [BN, C]
            current_features = current_features.view(B, N, self.channels)  # [B, N, C]
            
            # Prepare for next layer: add spatial coords back
            if layer != self.message_passing[-1]:  # Not last layer
                current_features = torch.cat([
                    current_features,
                    node_features[:, :, -2:]  # spatial coords
                ], dim=2)  # [B, N, C+2]
        
        return current_features  # [B, N, C]
    
    def _reproject_features(
        self,
        enhanced_features: torch.Tensor,
        node_indices: torch.Tensor,
        original_shape: Tuple[int, int, int, int]
    ) -> torch.Tensor:
        """
        Re-project enhanced node features back to original feature map resolution.
        
        Args:
            enhanced_features: Enhanced node features [B, N, C]
            node_indices: Spatial indices of nodes [B, N, 2] (y, x)
            original_shape: Original feature map shape (B, C, H, W)
            
        Returns:
            reprojected: Re-projected feature map [B, C, H, W]
        """
        B, N, C = enhanced_features.shape
        _, _, H, W = original_shape
        
        # Initialize output feature map
        reprojected = torch.zeros(B, C, H, W, device=enhanced_features.device, dtype=enhanced_features.dtype)
        
        # Scatter node features back to spatial locations
        batch_indices = torch.arange(B, device=enhanced_features.device).view(B, 1, 1).expand(-1, N, 1)
        node_y = node_indices[:, :, 0].long()  # [B, N]
        node_x = node_indices[:, :, 1].long()   # [B, N]
        
        # Clamp indices to valid range
        node_y = torch.clamp(node_y, 0, H - 1)
        node_x = torch.clamp(node_x, 0, W - 1)
        
        # Scatter features: use max aggregation for overlapping nodes
        # Completely non-inplace approach using scatter_reduce
        # Flatten spatial dimensions for easier indexing
        flat_indices = node_y * W + node_x  # [B, N]
        
        # Build reprojected tensor channel by channel (completely non-inplace)
        reprojected_list = []
        for c in range(C):
            channel_tensor = torch.zeros(B, H * W, device=enhanced_features.device, dtype=enhanced_features.dtype)
            
            for b in range(B):
                # Get features for this batch and channel
                node_feat = enhanced_features[b, :, c]  # [N]
                indices = flat_indices[b]  # [N]
                
                # Use scatter_reduce with max reduction (non-inplace)
                scattered = torch.zeros(H * W, device=enhanced_features.device, dtype=enhanced_features.dtype)
                scattered = scattered.scatter_reduce(0, indices, node_feat, reduce='amax', include_self=False)
                channel_tensor[b] = scattered
            
            reprojected_list.append(channel_tensor.view(B, 1, H, W))
        
        # Concatenate all channels (completely non-inplace)
        reprojected = torch.cat(reprojected_list, dim=1)  # [B, C, H, W]
        
        # Apply reprojection convolution
        reprojected = self.reproject(reprojected)
        
        return reprojected
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through connectivity-aware module.
        
        This module:
        1. Extracts high-activation locations as graph nodes
        2. Builds KNN graph (non-learnable) based on feature + spatial distance
        3. Performs message passing to learn contour membership
        4. Re-projects enhanced features back to spatial grid
        5. Fuses with original features via residual connection
        
        Args:
            x: Input feature map [B, C, H, W]
            mask: Optional binary mask [B, H, W] for node extraction
            
        Returns:
            output: Enhanced feature map [B, C, H, W] with structural connectivity awareness
        """
        B, C, H, W = x.shape
        
        # Step 1: Extract high-activation nodes
        node_features, node_coords, node_indices = self._extract_nodes(x, mask)  # [B, N, C+2], [B, N, 2], [B, N, 2]
        
        # Step 2: Build KNN graph (non-learnable)
        adjacency = self._build_knn_graph(node_coords, node_features)  # [B, N, N]
        
        # Step 3: Connectivity-aware message passing
        enhanced_features = self._message_passing(node_features, adjacency)  # [B, N, C]
        
        # Step 4: Re-project enhanced features back to spatial grid
        reprojected = self._reproject_features(enhanced_features, node_indices, (B, C, H, W))  # [B, C, H, W]
        
        # Step 5: Fuse with original features (residual connection)
        # This allows the network to learn whether to use graph-enhanced or original features
        output = x + reprojected
        
        return output


class ConnectivityBlock(nn.Module):
    """
    Connectivity Block: Wraps connectivity-aware module with standard convolution.
    
    This block can be inserted into YOLO backbone to enable structural contour
    formation from sparse event points before bounding box regression.
    """
    
    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        num_nodes: int = 128,
        k_neighbors: int = 8,
        message_passing_layers: int = 2,
    ):
        """
        Initialize Connectivity Block.
        
        Args:
            c1 (int): Input channels
            c2 (int): Output channels
            k (int): Convolution kernel size
            s (int): Convolution stride
            num_nodes (int): Maximum number of nodes to extract
            k_neighbors (int): Number of neighbors for KNN graph
            message_passing_layers (int): Number of GNN layers
        """
        super().__init__()
        
        # Standard convolution
        from .conv import Conv
        self.conv = Conv(c1, c2, k, s, act=True)
        
        # Connectivity-aware module (only if channels match and no downsampling)
        if c1 == c2 and s == 1:
            self.connectivity = ConnectivityAwareModule(
                c2, 
                num_nodes=num_nodes,
                k_neighbors=k_neighbors,
                message_passing_layers=message_passing_layers
            )
        else:
            self.connectivity = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: convolution -> connectivity-aware enhancement.
        
        Args:
            x: Input feature map [B, C, H, W]
            
        Returns:
            output: Enhanced feature map with structural connectivity awareness
        """
        x = self.conv(x)
        x = self.connectivity(x)
        return x


def ConnectivityAwareBlock(
    c1: int,
    c2: int,
    k: int = 3,
    s: int = 1,
    num_nodes: int = 128,
    k_neighbors: int = 8,
    message_passing_layers: int = 2,
    **kwargs
) -> ConnectivityBlock:
    """
    Factory function to create Connectivity Block.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        k (int): Convolution kernel size
        s (int): Convolution stride
        num_nodes (int): Maximum number of nodes to extract
        k_neighbors (int): Number of neighbors for KNN graph
        message_passing_layers (int): Number of GNN layers
        **kwargs: Additional arguments (ignored)
        
    Returns:
        ConnectivityBlock: Connectivity-aware block instance
    """
    return ConnectivityBlock(
        c1, c2, k, s,
        num_nodes=num_nodes,
        k_neighbors=k_neighbors,
        message_passing_layers=message_passing_layers
    )
