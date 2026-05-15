# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
BiFormer Neck module for YOLO
集成 BiFormer 的 Bi-Level Routing Attention 到 YOLO neck
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加项目路径以导入 BiFormer
# ultralytics 可能安装在 site-packages 中，需要向上查找项目根目录
current_file = Path(__file__).resolve()
# 从 ultralytics/ultralytics/nn/modules/biformer.py 向上查找到项目根目录
# 可能的路径：project_root/ultralytics/ultralytics/nn/modules/biformer.py
# 或：site-packages/ultralytics/nn/modules/biformer.py
possible_roots = [
    current_file.parent.parent.parent.parent.parent,  # project_root/ultralytics/...
    current_file.parent.parent.parent.parent.parent.parent,  # 再上一级
]

biformer_path = None
for root in possible_roots:
    test_path = root / "BiFormer"
    if test_path.exists() and (test_path / "models" / "biformer.py").exists():
        biformer_path = test_path
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(biformer_path))
        break

# 如果没找到，尝试从环境变量或当前工作目录查找
if biformer_path is None or not biformer_path.exists():
    import os
    cwd = Path(os.getcwd())
    test_path = cwd / "BiFormer"
    if test_path.exists() and (test_path / "models" / "biformer.py").exists():
        biformer_path = test_path
        sys.path.insert(0, str(cwd))
        sys.path.insert(0, str(biformer_path))

# 导入 BiFormer 模块
try:
    from models.biformer import Block
    from ops.bra_legacy import BiLevelRoutingAttention
except ImportError:
    try:
        from BiFormer.models.biformer import Block
        from BiFormer.ops.bra_legacy import BiLevelRoutingAttention
    except ImportError:
        try:
            import importlib.util
            biformer_file = biformer_path / "models" / "biformer.py"
            if biformer_file.exists():
                spec = importlib.util.spec_from_file_location("biformer", biformer_file)
                biformer_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(biformer_module)
                Block = biformer_module.Block
                
                bra_file = biformer_path / "ops" / "bra_legacy.py"
                if bra_file.exists():
                    spec = importlib.util.spec_from_file_location("bra_legacy", bra_file)
                    bra_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(bra_module)
                    BiLevelRoutingAttention = bra_module.BiLevelRoutingAttention
            else:
                raise ImportError("找不到 BiFormer 模块")
        except Exception as e:
            raise ImportError(f"无法导入 BiFormer 模块: {e}")


class BiFormerBlock(nn.Module):
    """
    BiFormer Block 适配器，用于 ultralytics YOLO
    将 NCHW 格式的特征图通过 BiFormer Block 处理
    """
    def __init__(self, 
                 c1,  # 输入通道数
                 c2,  # 输出通道数
                 num_heads=8,
                 n_win=7,
                 topk=4,
                 mlp_ratio=4,
                 drop_path=0.0,
                 auto_pad=True,
                 **kwargs):
        """
        Args:
            c1: 输入通道数
            c2: 输出通道数
            num_heads: 注意力头数
            n_win: 窗口大小
            topk: BiFormer 路由的 topk 值
            mlp_ratio: MLP 扩展比例
            drop_path: DropPath 比率
            auto_pad: 是否自动 padding（支持任意尺寸输入）
        """
        super().__init__()
        
        # 如果通道数不同，先进行通道调整
        if c1 != c2:
            self.channel_adapter = nn.Sequential(
                nn.Conv2d(c1, c2, 1),
                nn.BatchNorm2d(c2)
            )
        else:
            self.channel_adapter = nn.Identity()
        
        # BiFormer Block
        self.block = Block(
            dim=c2,
            drop_path=drop_path,
            num_heads=num_heads,
            n_win=n_win,
            topk=topk,
            mlp_ratio=mlp_ratio,
            auto_pad=auto_pad,
            **kwargs
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C1, H, W) tensor
        Returns:
            x: (B, C2, H, W) tensor
        """
        x = self.channel_adapter(x)
        x = self.block(x)
        return x


class BiFormerNeck(nn.Module):
    """
    BiFormer Neck for YOLO
    使用 BiFormer 的注意力机制增强 YOLO 的多尺度特征融合
    
    在 YOLO 的 head 部分使用，替换标准的 C3k2 等模块
    """
    def __init__(self,
                 c1,  # 输入通道数
                 c2,  # 输出通道数
                 num_heads=8,
                 n_win=7,
                 topk=4,
                 mlp_ratio=4,
                 drop_path=0.0,
                 auto_pad=True,
                 **kwargs):
        """
        Args:
            c1: 输入通道数
            c2: 输出通道数
            num_heads: 注意力头数
            n_win: 窗口大小
            topk: BiFormer 路由的 topk 值
            mlp_ratio: MLP 扩展比例
            drop_path: DropPath 比率
            auto_pad: 是否自动 padding
        """
        super().__init__()
        self.biformer_block = BiFormerBlock(
            c1=c1,
            c2=c2,
            num_heads=num_heads,
            n_win=n_win,
            topk=topk,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            auto_pad=auto_pad,
            **kwargs
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C1, H, W) tensor 或 list of tensors（多尺度输入）
        Returns:
            x: (B, C2, H, W) tensor
        """
        # 如果是列表（多尺度输入），处理第一个
        if isinstance(x, (list, tuple)):
            x = x[0] if len(x) > 0 else x
        
        return self.biformer_block(x)
