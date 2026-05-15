import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
project_root = Path(__file__).parent.parent
biformer_path = project_root / "BiFormer"
if biformer_path.exists():
    sys.path.insert(0, str(biformer_path))
    sys.path.insert(0, str(project_root))
Block = None
BiLevelRoutingAttention = None
try:
    from models.biformer import Block
    from ops.bra_legacy import BiLevelRoutingAttention
    print(f"[BiFormerNeck] 从 BiFormer/models 导入成功")
except ImportError:
    try:
        from BiFormer.models.biformer import Block
        from BiFormer.ops.bra_legacy import BiLevelRoutingAttention
        print(f"[BiFormerNeck] 从 BiFormer/ 导入成功")
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
                print(f"[BiFormerNeck] 从文件直接导入成功")
            else:
                raise ImportError("找不到 BiFormer 模块文件")
        except Exception as e:
            raise ImportError(f"无法导入 BiFormer 模块: {e}\n请检查 BiFormer 路径: {biformer_path}")
if Block is None or BiLevelRoutingAttention is None:
    raise ImportError("BiFormer 模块导入失败，请检查路径和依赖")
class BiFormerNeckBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 n_win=7,
                 topk=4,
                 mlp_ratio=4,
                 drop_path=0.0,
                 auto_pad=True,
                 **kwargs):
        super().__init__()
        self.block = Block(
            dim=dim,
            drop_path=drop_path,
            num_heads=num_heads,
            n_win=n_win,
            topk=topk,
            mlp_ratio=mlp_ratio,
            auto_pad=auto_pad,
            **kwargs
        )
    def forward(self, x):
        return self.block(x)
class BiFormerNeck(nn.Module):
    def __init__(self,
                 in_channels=[256, 512, 1024],
                 out_channels=[256, 512, 1024],
                 num_heads=[8, 8, 8],
                 n_win=7,
                 topk=4,
                 mlp_ratio=4,
                 drop_path=0.0,
                 auto_pad=True,
                 use_biformer=True):
        super().__init__()
        assert len(in_channels) == len(out_channels),\
            f"输入和输出通道数列表长度必须相同: {len(in_channels)} != {len(out_channels)}"
        self.num_scales = len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_biformer = use_biformer
        self.enhance_modules = nn.ModuleList()
        for i, (in_c, out_c) in enumerate(zip(in_channels, out_channels)):
            if use_biformer:
                if in_c != out_c:
                    self.enhance_modules.append(
                        nn.Sequential(
                            nn.Conv2d(in_c, out_c, 1),
                            nn.BatchNorm2d(out_c),
                            BiFormerNeckBlock(
                                dim=out_c,
                                num_heads=num_heads[i] if isinstance(num_heads, list) else num_heads,
                                n_win=n_win,
                                topk=topk,
                                mlp_ratio=mlp_ratio,
                                drop_path=drop_path,
                                auto_pad=auto_pad
                            )
                        )
                    )
                else:
                    self.enhance_modules.append(
                        BiFormerNeckBlock(
                            dim=out_c,
                            num_heads=num_heads[i] if isinstance(num_heads, list) else num_heads,
                            n_win=n_win,
                            topk=topk,
                            mlp_ratio=mlp_ratio,
                            drop_path=drop_path,
                            auto_pad=auto_pad
                        )
                    )
            else:
                self.enhance_modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_c, out_c, 1),
                        nn.BatchNorm2d(out_c),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_c, out_c, 3, padding=1),
                        nn.BatchNorm2d(out_c),
                        nn.ReLU(inplace=True)
                    )
                )
        self.top_down_fusion = nn.ModuleList()
        for i in range(self.num_scales - 1):
            self.top_down_fusion.append(
                nn.Sequential(
                    nn.Conv2d(out_channels[i+1], out_channels[i], 1),
                    nn.BatchNorm2d(out_channels[i]),
                    nn.ReLU(inplace=True)
                )
            )
        self.bottom_up_fusion = nn.ModuleList()
        for i in range(self.num_scales - 1):
            self.bottom_up_fusion.append(
                nn.Sequential(
                    nn.Conv2d(out_channels[i], out_channels[i+1], 3, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels[i+1]),
                    nn.ReLU(inplace=True)
                )
            )
    def forward(self, features):
        assert len(features) == self.num_scales,\
            f"输入特征数量 ({len(features)}) 与配置的尺度数量 ({self.num_scales}) 不匹配"
        enhanced = []
        for i, feat in enumerate(features):
            enhanced_feat = self.enhance_modules[i](feat)
            enhanced.append(enhanced_feat)
        top_down = [enhanced[-1]]
        for i in range(self.num_scales - 2, -1, -1):
            upsampled = F.interpolate(
                top_down[0],
                size=enhanced[i].shape[2:],
                mode='nearest'
            )
            fused = self.top_down_fusion[i](upsampled)
            top_down.insert(0, enhanced[i] + fused)
        outputs = [top_down[0]]
        for i in range(self.num_scales - 1):
            downsampled = self.bottom_up_fusion[i](outputs[-1])
            target_size = top_down[i+1].shape[2:]
            if downsampled.shape[2:] != target_size:
                downsampled = F.interpolate(
                    downsampled,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
            outputs.append(top_down[i+1] + downsampled)
        return outputs
def create_biformer_neck(in_channels=[256, 512, 1024],
                         out_channels=[256, 512, 1024],
                         **kwargs):
    return BiFormerNeck(
        in_channels=in_channels,
        out_channels=out_channels,
        **kwargs
    )
