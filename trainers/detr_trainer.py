
import os
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime
import yaml
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
detr_path = project_root / "detr-main"
sys.path.insert(0, str(detr_path))
from utils.metrics_logger import MetricsLogger
class DETRTrainer:
    SUPPORTED_BACKBONES = {
        "resnet18": {"params": 11, "flops": 15},
        "resnet34": {"params": 21, "flops": 25},
        "resnet50": {"params": 41, "flops": 86},
        "resnet101": {"params": 60, "flops": 152},
    }
    def __init__(self,
                 model_name: str = "detr",
                 backbone: str = "resnet50",
                 data_yaml: str = None,
                 project_name: str = "fish_detection",
                 experiment_name: str = None,
                 device: str = "auto",
                 pretrained: bool = False):
        self.model_name = model_name.lower()
        self.backbone = backbone.lower()
        self.data_yaml = data_yaml
        self.project_name = project_name
        self.pretrained = pretrained
        self.num_classes = 1
        self.coco_path = None
        self.class_names = ['fish']
        if data_yaml:
            self._parse_data_yaml(data_yaml)
        pretrained_tag = "pretrained" if pretrained else "scratch"
        self.experiment_name = experiment_name or f"{model_name}_{backbone}_{pretrained_tag}"
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = None
        self.criterion = None
        self.postprocessors = None
        self.save_dir = project_root / "results" / self.project_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = Path(data_yaml).stem if data_yaml else "unknown"
        self.logger = MetricsLogger(
            experiment_name=self.experiment_name,
            model_name=f"DETR-{self.backbone}",
            dataset_name=dataset_name,
            save_dir=str(self.save_dir)
        )
    def _parse_data_yaml(self, data_yaml: str):
        with open(data_yaml, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        nc = config.get('nc', 1)
        try:
            import json
            coco_format_path = Path(config.get('path', ''))
            if not coco_format_path.is_absolute():
                coco_format_path = project_root / coco_format_path
            coco_format_path = coco_format_path / "COCO_format"
            train_ann = coco_format_path / "annotations" / "instances_train2017.json"
            if train_ann.exists():
                with open(train_ann, 'r', encoding='utf-8') as f:
                    coco_data = json.load(f)
                max_category_id = max([c['id'] for c in coco_data['categories']]) if coco_data.get('categories') else nc
                self.num_classes = max_category_id + 1
                print(f"✓ 从COCO数据检测到max category_id={max_category_id}，设置num_classes={self.num_classes}")
                print(f"  说明：DETR的class_embed将创建{self.num_classes+1}个输出（类别0到{self.num_classes}）")
                print(f"  其中：类别0={max_category_id-1}对应COCO的category_id={max_category_id}，类别{self.num_classes}是no-object")
            else:
                self.num_classes = nc + 1
                print(f"⚠️  COCO数据不存在，使用默认值：num_classes={self.num_classes} (假设category_id从1开始)")
        except Exception as e:
            self.num_classes = nc + 1
            print(f"⚠️  检测COCO数据失败: {e}，使用默认值：num_classes={self.num_classes}")
        self.class_names = config.get('names', ['fish'])
        print(f"类别数: {self.num_classes}, 类别: {self.class_names}")
        print(f"⚠️  注意：DETR期望类别索引从0开始，需要COCO的category_id减1")
        base_path = Path(config.get('path', ''))
        if not base_path.is_absolute():
            base_path = project_root / base_path
        coco_format_path = base_path / "COCO_format"
        if coco_format_path.exists():
            self.coco_path = str(coco_format_path)
            print(f"✓ 找到COCO格式数据: {self.coco_path}")
        else:
            self.coco_path = str(base_path)
            print(f"⚠️  未找到COCO_format目录")
            print(f"⚠️  请先运行: python convert_all_to_coco.py")
    def _get_args(self, **kwargs):
        from argparse import Namespace
        args = Namespace(
            lr=kwargs.get('lr', 1e-4),
            lr_backbone=kwargs.get('lr_backbone', 1e-5),
            batch_size=kwargs.get('batch_size', 2),
            weight_decay=kwargs.get('weight_decay', 1e-4),
            epochs=kwargs.get('epochs', 300),
            lr_drop=kwargs.get('lr_drop', 200),
            clip_max_norm=kwargs.get('clip_max_norm', 0.1),

            frozen_weights=None,
            backbone=self.backbone,
            dilation=False,
            position_embedding='sine',

            enc_layers=kwargs.get('enc_layers', 6),
            dec_layers=kwargs.get('dec_layers', 6),
            dim_feedforward=kwargs.get('dim_feedforward', 2048),
            hidden_dim=kwargs.get('hidden_dim', 256),
            dropout=kwargs.get('dropout', 0.1),
            nheads=kwargs.get('nheads', 8),
            num_queries=kwargs.get('num_queries', 100),
            pre_norm=False,

            masks=False,
            aux_loss=True,
            set_cost_class=1,
            set_cost_bbox=5,
            set_cost_giou=2,
            mask_loss_coef=1,
            dice_loss_coef=1,
            bbox_loss_coef=5,
            giou_loss_coef=2,
            eos_coef=0.2,

            dataset_file='coco',
            coco_path=self.coco_path,
            coco_panoptic_path=None,
            remove_difficult=False,

            output_dir=str(self.save_dir / "runs" / self.experiment_name),
            device=self.device,
            seed=42,
            resume='',
            start_epoch=0,
            eval=False,
            num_workers=0,
            distributed=False,
            world_size=1,
            num_classes=self.num_classes,
        )
        return args
    def load_model(self, pretrained: bool = None):
        if pretrained is None:
            pretrained = self.pretrained
        try:
            from models import build_model
        except ImportError:
            sys.path.insert(0, str(detr_path))
            from models import build_model
        args = self._get_args()
        print(f"加载DETR模型 (backbone: {self.backbone})")
        print(f"训练模式: {'使用预训练backbone' if pretrained else '完全从头训练'}")
        self.model, self.criterion, self.postprocessors = build_model(args)
        if pretrained:
            print("\n" + "="*60)
            print("开始加载预训练权重...")
            print("="*60)
            if not self._load_pretrained_detr():
                print("⚠️  无法加载完整DETR预训练权重，改为加载ImageNet预训练的backbone权重")
                self._load_pretrained_backbone()
            print("="*60 + "\n")
        else:
            self._reinit_backbone()
            print("✓ DETR支持完全从头训练，适合科研公平对比")
        self.model.to(self.device)
        self._log_model_info()
        return self.model
    def _load_pretrained_backbone(self):
        import torchvision.models as models
        print(f"正在加载ImageNet预训练的{self.backbone}权重...")
        if self.backbone == "resnet18":
            pretrained_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif self.backbone == "resnet34":
            pretrained_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        elif self.backbone == "resnet50":
            pretrained_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        elif self.backbone == "resnet101":
            pretrained_model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        else:
            print(f"⚠️ 不支持的backbone: {self.backbone}，跳过预训练加载")
            return
        try:
            if hasattr(self.model, 'backbone') and len(self.model.backbone) > 0:
                backbone_module = self.model.backbone[0]
                resnet_backbone = None
                if hasattr(backbone_module.body, '_modules'):
                    modules = backbone_module.body._modules
                    for key, module in modules.items():
                        if isinstance(module, torch.nn.Module):
                            if hasattr(module, 'conv1') and hasattr(module, 'layer1'):
                                resnet_backbone = module
                                print(f"  找到ResNet backbone在: body.{key}")
                                break
                if resnet_backbone is None:
                    detr_state_dict = backbone_module.body.state_dict()
                    pretrained_dict = pretrained_model.state_dict()
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'fc' not in k}
                    matched_dict = {}
                    for pretrained_key, pretrained_value in pretrained_dict.items():
                        if pretrained_key in detr_state_dict:
                            if detr_state_dict[pretrained_key].shape == pretrained_value.shape:
                                matched_dict[pretrained_key] = pretrained_value
                    if len(matched_dict) > 0:
                        detr_state_dict.update(matched_dict)
                        backbone_module.body.load_state_dict(detr_state_dict, strict=False)
                        print(f"✓ 通过state_dict匹配加载了 {len(matched_dict)}/{len(pretrained_dict)} 个backbone权重")
                        if 'conv1.weight' in matched_dict:
                            updated_dict = backbone_module.body.state_dict()
                            if 'conv1.weight' in updated_dict:
                                weight_diff = (updated_dict['conv1.weight'] - pretrained_model.conv1.weight.data).abs().mean().item()
                                if weight_diff < 1e-5:
                                    print(f"✓ 验证通过：backbone第一层权重匹配（差异: {weight_diff:.2e}）")
                                else:
                                    print(f"⚠️ 警告：backbone第一层权重不匹配（差异: {weight_diff:.2e}）")
                        return
                if resnet_backbone is not None:
                    pretrained_dict = pretrained_model.state_dict()
                    model_dict = resnet_backbone.state_dict()
                    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                                     if k in model_dict and 'fc' not in k}
                    model_dict.update(pretrained_dict)
                    resnet_backbone.load_state_dict(model_dict, strict=False)
                    if 'conv1.weight' in pretrained_dict:
                        weight_diff = (resnet_backbone.conv1.weight.data - pretrained_model.conv1.weight.data).abs().mean().item()
                        if weight_diff < 1e-6:
                            print(f"✓ 验证通过：backbone第一层权重匹配（差异: {weight_diff:.2e}）")
                        else:
                            print(f"⚠️ 警告：backbone第一层权重不匹配（差异: {weight_diff:.2e}）")
                    print(f"✓ 成功加载 {len(pretrained_dict)}/{len(model_dict)} 个backbone权重")
                    return
                print("⚠️ 无法找到ResNet backbone或匹配权重")
                print(f"  body类型: {type(backbone_module.body)}")
                print(f"  body属性: {[attr for attr in dir(backbone_module.body) if not attr.startswith('__')][:10]}")
                if hasattr(backbone_module.body, '_modules'):
                    print(f"  body._modules键: {list(backbone_module.body._modules.keys())}")
            else:
                print("⚠️ 模型没有backbone属性，无法加载预训练权重")
        except Exception as e:
            print(f"⚠️ 加载预训练权重失败: {e}")
            import traceback
            traceback.print_exc()
    def _load_pretrained_detr(self):
        import urllib.request
        detr_pretrained_urls = {
            "resnet50": "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth",
            "resnet101": "https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth",
        }
        if self.backbone not in detr_pretrained_urls:
            print(f"⚠️  DETR官方只提供ResNet50和ResNet101的预训练权重，当前backbone={self.backbone}")
            return False
        try:
            print(f"正在下载DETR预训练权重（{self.backbone}）...")
            url = detr_pretrained_urls[self.backbone]
            checkpoint_path = project_root / f"detr_{self.backbone}_pretrained.pth"
            if not checkpoint_path.exists():
                print(f"  从 {url} 下载...")
                urllib.request.urlretrieve(url, checkpoint_path)
                print(f"  ✓ 下载完成: {checkpoint_path}")
            else:
                print(f"  ✓ 使用已存在的权重文件: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            pretrained_state_dict = checkpoint.get('model', checkpoint)
            model_state_dict = self.model.state_dict()
            filtered_state_dict = {}
            skipped_keys = []
            for key, value in pretrained_state_dict.items():
                if 'class_embed' in key or 'bbox_embed' in key:
                    skipped_keys.append(key)
                    continue
                if key in model_state_dict:
                    if model_state_dict[key].shape == value.shape:
                        filtered_state_dict[key] = value
                    else:
                        skipped_keys.append(f"{key} (shape mismatch)")
                else:
                    skipped_keys.append(f"{key} (not found)")
            model_state_dict.update(filtered_state_dict)
            self.model.load_state_dict(model_state_dict, strict=False)
            print(f"✓ 成功加载 {len(filtered_state_dict)}/{len(pretrained_state_dict)} 个DETR权重")
            if skipped_keys:
                print(f"  跳过 {len(skipped_keys)} 个权重（分类头/bbox头，因为类别数不同）")
            return True
        except Exception as e:
            print(f"⚠️  加载DETR预训练权重失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    def _reinit_backbone(self):
        print("正在重新初始化backbone权重...")
        def init_weights(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
        if hasattr(self.model, 'backbone'):
            self.model.backbone.apply(init_weights)
            print("✓ Backbone权重已重新初始化")
    def _log_model_info(self):
        if self.model is None:
            return
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        backbone_info = self.SUPPORTED_BACKBONES.get(self.backbone, {"params": 41, "flops": 86})
        flops = backbone_info["flops"]
        model_size = total_params * 4 / (1024 * 1024)
        self.logger.set_model_info(
            params=total_params,
            flops=flops,
            model_size_mb=model_size,
            input_size=(640, 640)
        )
        print(f"模型参数量: {total_params/1e6:.2f}M")
        print(f"可训练参数: {trainable_params/1e6:.2f}M")
        print(f"估算FLOPs: {flops}G")
        print(f"模型大小: {model_size:.2f}MB")
    def train(self,
              epochs: int = 100,
              batch_size: int = 1,
              imgsz: int = 640,
              lr0: float = 2e-4,
              lr_backbone: float = 2e-4,
              weight_decay: float = 1e-4,
              lr_drop: int = 80,
              warmup_epochs: int = 10,
              patience: int = 50,
              save_period: int = 10,
              workers: int = 0,
              amp: bool = False,
              resume: bool = False,
              **kwargs) -> Dict:
        if self.model is None:
            self.load_model()
        if self.data_yaml is None:
            raise ValueError("请指定数据集配置文件 (data_yaml)")
        config = {
            "model": f"DETR-{self.backbone}",
            "backbone": self.backbone,
            "pretrained": self.pretrained,
            "epochs": epochs,
            "batch_size": batch_size,
            "imgsz": imgsz,
            "lr0": lr0,
            "lr_backbone": lr_backbone,
            "weight_decay": weight_decay,
            "lr_drop": lr_drop,
            "device": self.device,
            "num_classes": self.num_classes,
        }
        self.logger.set_config(config)
        print(f"\n{'='*60}")
        print(f"开始训练: DETR ({self.backbone})")
        print(f"数据集: {self.data_yaml}")
        print(f"设备: {self.device}")
        print(f"训练轮数: {epochs}")
        print(f"批次大小: {batch_size}")
        print(f"从头训练: {'否' if self.pretrained else '是'}")
        print(f"{'='*60}\n")
        start_time = time.time()
        best_map = 0.0
        try:
            from engine import train_one_epoch, evaluate
            from datasets import build_dataset, get_coco_api_from_dataset
            from torch.utils.data import DataLoader
            import util.misc as utils
            args = self._get_args(
                epochs=epochs,
                batch_size=batch_size,
                lr=lr0,
                lr_backbone=lr_backbone,
                weight_decay=weight_decay,
                lr_drop=lr_drop,
            )
            coco_path = Path(self.coco_path)
            train_ann = coco_path / "annotations" / "instances_train2017.json"
            if not train_ann.exists():
                raise FileNotFoundError(
                    f"COCO数据不完整！请先运行: python convert_all_to_coco.py"
                )
            print(f"✓ COCO数据验证通过")
            dataset_train = build_dataset(image_set='train', args=args)
            dataset_val = build_dataset(image_set='val', args=args)
            print(f"训练集: {len(dataset_train)}张图像")
            print(f"验证集: {len(dataset_val)}张图像")
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train, batch_size, drop_last=True)
            data_loader_train = DataLoader(
                dataset_train, batch_sampler=batch_sampler_train,
                collate_fn=utils.collate_fn, num_workers=workers,
                pin_memory=False
            )
            data_loader_val = DataLoader(
                dataset_val, batch_size=batch_size,
                sampler=sampler_val, drop_last=False,
                collate_fn=utils.collate_fn, num_workers=workers,
                pin_memory=False
            )
            param_dicts = [
                {"params": [p for n, p in self.model.named_parameters()
                           if "backbone" not in n and p.requires_grad],
                 "initial_lr": lr0},
                {"params": [p for n, p in self.model.named_parameters()
                           if "backbone" in n and p.requires_grad],
                 "lr": lr_backbone,
                 "initial_lr": lr_backbone},
            ]
            optimizer = torch.optim.AdamW(param_dicts, lr=lr0, weight_decay=weight_decay)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - warmup_epochs, eta_min=lr0 * 0.01
            )
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n开始训练，共{epochs}个epoch...")
            for epoch in range(epochs):
                if epoch < warmup_epochs:
                    warmup_factor = (epoch + 1) / warmup_epochs
                    for param_group in optimizer.param_groups:
                        initial_lr = param_group.get('initial_lr', param_group.get('lr', lr0))
                        param_group['lr'] = initial_lr * warmup_factor
                else:
                    lr_scheduler.step()
                train_stats = train_one_epoch(
                    self.model, self.criterion, data_loader_train, optimizer,
                    self.device, epoch, args.clip_max_norm
                )
                train_loss = train_stats.get('loss', 0)
                loss_ce = train_stats.get('loss_ce', 0)
                loss_bbox = train_stats.get('loss_bbox', 0)
                loss_giou = train_stats.get('loss_giou', 0)
                class_error = train_stats.get('class_error', 100.0)
                cardinality_error = train_stats.get('cardinality_error', 0.0)
                if epoch < 3:
                    print(f"\n[Epoch {epoch+1} 调试信息]")
                    print(f"  class_error: {class_error:.2f}%")
                    print(f"  cardinality_error: {cardinality_error:.4f}")
                    print(f"  loss_ce: {loss_ce:.4f}")
                    print(f"  backbone LR: {optimizer.param_groups[1]['lr']:.2e}")
                    print(f"  transformer LR: {optimizer.param_groups[0]['lr']:.2e}")
                    self.model.eval()
                    with torch.no_grad():
                        sample_loader = iter(data_loader_train)
                        samples, targets = next(sample_loader)
                        samples = samples.to(self.device)
                        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                        outputs = self.model(samples)
                        pred_logits = outputs['pred_logits']
                        pred_probs = pred_logits.softmax(-1)
                        pred_classes = pred_logits.argmax(-1)
                        num_queries = pred_classes.shape[1]
                        num_batch = pred_classes.shape[0]
                        class_counts = {}
                        for cls_id in range(self.num_classes + 1):
                            count = (pred_classes == cls_id).sum().item()
                            class_counts[cls_id] = count
                        print(f"  预测类别分布 (总查询数: {num_queries * num_batch}):")
                        print(f"  注意：DETR的class_embed输出维度={self.num_classes + 1}（类别0到{self.num_classes}）")
                        for cls_id in range(self.num_classes + 1):
                            if cls_id < len(self.class_names):
                                coco_cat_id = cls_id + 1
                                cls_name = f"{self.class_names[cls_id]} (COCO cat_id={coco_cat_id})"
                            elif cls_id == self.num_classes:
                                cls_name = "no-object (背景类)"
                            else:
                                cls_name = f"empty (类别{cls_id})"
                            count = class_counts[cls_id]
                            pct = count / (num_queries * num_batch) * 100
                            print(f"    {cls_id}: {count} ({pct:.1f}%) - {cls_name}")
                        indices = self.criterion.matcher(outputs, targets)
                        total_matched = sum(len(idx[0]) for idx in indices)
                        print(f"  匹配到的预测框数量: {total_matched} (GT总数: {sum(len(t['labels']) for t in targets)})")
                        if total_matched > 0:
                            matched_pred_classes = []
                            for i, (src_idx, tgt_idx) in enumerate(indices):
                                if len(src_idx) > 0:
                                    batch_pred_classes = pred_classes[i][src_idx]
                                    matched_pred_classes.extend(batch_pred_classes.cpu().tolist())
                            matched_class_counts = {}
                            for cls_id in range(self.num_classes + 1):
                                count = matched_pred_classes.count(cls_id)
                                matched_class_counts[cls_id] = count
                            print(f"  匹配到的预测框类别分布:")
                            for cls_id in range(self.num_classes + 1):
                                if cls_id < len(self.class_names):
                                    coco_cat_id = cls_id + 1
                                    cls_name = f"{self.class_names[cls_id]} (COCO cat_id={coco_cat_id})"
                                elif cls_id == self.num_classes:
                                    cls_name = "no-object (背景类)"
                                else:
                                    cls_name = f"empty (类别{cls_id})"
                                count = matched_class_counts.get(cls_id, 0)
                                pct = count / total_matched * 100 if total_matched > 0 else 0
                                print(f"    {cls_id}: {count} ({pct:.1f}%) - {cls_name}")
                    self.model.train()
                self.logger.log_train_loss(
                    epoch=epoch + 1,
                    loss=train_loss,
                    box_loss=loss_bbox,
                    cls_loss=loss_ce,
                    giou_loss=loss_giou
                )
                if (epoch + 1) % save_period == 0 or epoch == epochs - 1:
                    base_ds = get_coco_api_from_dataset(dataset_val)
                    test_stats, coco_evaluator = evaluate(
                        self.model, self.criterion, self.postprocessors,
                        data_loader_val, base_ds, self.device, str(output_dir)
                    )
                    if coco_evaluator is not None:
                        coco_eval = coco_evaluator.coco_eval['bbox']
                        stats = coco_eval.stats
                        self.logger.log_epoch_metrics(
                            epoch=epoch + 1,
                            precision=stats[0],
                            recall=stats[8] if len(stats) > 8 else 0,
                            map50=stats[1],
                            map50_95=stats[0]
                        )
                        if stats[0] > best_map:
                            best_map = stats[0]
                            torch.save({
                                'model': self.model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'epoch': epoch,
                                'best_map': best_map,
                            }, output_dir / 'best.pth')
                            print(f"  ✓ 保存最佳模型 (mAP: {best_map:.4f})")
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"\n训练完成! 最佳mAP@0.5:0.95: {best_map:.4f}")
        except Exception as e:
            print(f"训练出错: {e}")
            import traceback
            traceback.print_exc()
            raise
        training_time = time.time() - start_time
        print(f"训练用时: {training_time/3600:.2f}小时")
        return {
            "model_name": f"DETR-{self.backbone}",
            "training_time": training_time,
            "best_map": best_map,
            "best_metrics": self.logger.get_best_metrics(),
        }
    def benchmark_inference(self, imgsz: int = 640, warmup: int = 10, runs: int = 100):
        if self.model is None:
            self.load_model()
        self.model.eval()
        dummy_input = torch.rand(1, 3, imgsz, imgsz).to(self.device)
        print(f"推理性能测试 (预热{warmup}次, 测试{runs}次)...")
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(dummy_input)
        if self.device == "cuda":
            torch.cuda.synchronize()
        times = []
        with torch.no_grad():
            for _ in range(runs):
                start = time.perf_counter()
                _ = self.model(dummy_input)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1000 / avg_time
        self.logger.set_inference_metrics(
            inference_time_ms=avg_time,
            fps=fps,
            latency_ms=avg_time,
            batch_size=1,
            input_size=(imgsz, imgsz),
            device=self.device
        )
        print(f"推理时间: {avg_time:.2f}ms (±{std_time:.2f}ms)")
        print(f"最小/最大: {min_time:.2f}ms / {max_time:.2f}ms")
        print(f"FPS: {fps:.2f}")
        print(f"端到端延迟: {avg_time:.2f}ms")
        return {
            "inference_time_ms": avg_time,
            "fps": fps,
            "latency_ms": avg_time,
            "std_ms": std_time
        }
    def get_results(self) -> Dict:
        return {
            "model_name": f"DETR-{self.backbone}",
            "experiment_name": self.experiment_name,
            "model_info": self.logger.model_info,
            "config": self.logger.config,
            "best_metrics": self.logger.get_best_metrics(),
            "inference_metrics": self.logger.inference_metrics,
            "save_dir": str(self.logger.exp_dir)
        }
    def save_results(self):
        excel_path, json_path = self.logger.save_all()
        print(f"\n结果已保存:")
        print(f"  Excel: {excel_path}")
        print(f"  JSON: {json_path}")
        return excel_path, json_path
if __name__ == "__main__":
    print("DETR训练器测试")
    print("="*60)
    trainer = DETRTrainer(
        backbone="resnet18",
        data_yaml="configs/dataset_blur_png.yaml",
        pretrained=False
    )
    trainer.load_model()
    trainer.benchmark_inference(warmup=3, runs=10)
    trainer.save_results()
    print("\n测试完成!")
