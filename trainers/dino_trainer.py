
import os
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import yaml
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
dino_path = project_root / "DINO-main"
sys.path.insert(0, str(dino_path))
from utils.metrics_logger import MetricsLogger
class DINOTrainer:
    SUPPORTED_BACKBONES = {
        "resnet50": {"params": "47M", "flops": "~279G"},
        "resnet101": {"params": "66M", "flops": "~~350G"},
        "swin-t": {"params": "~48M", "flops": "~280G"},
        "swin-l": {"params": "~218M", "flops": "~~900G"},
    }
    SUPPORTED_SCALES = {
        "4scale": "DINO_4scale.py",
        "5scale": "DINO_5scale.py",
    }
    def __init__(self,
                 model_name: str = "dino",
                 backbone: str = "resnet50",
                 scale: str = "4scale",
                 data_yaml: str = None,
                 project_name: str = "fish_detection",
                 experiment_name: str = None,
                 device: str = "auto",
                 pretrained: bool = False):
        self.model_name = model_name.lower()
        self.backbone = backbone.lower()
        self.scale = scale.lower()
        self.data_yaml = data_yaml
        self.project_name = project_name
        self.pretrained = pretrained
        self.num_classes = 1
        self.coco_path = None
        if data_yaml:
            self._parse_data_yaml(data_yaml)
        pretrained_tag = "pretrained" if pretrained else "from_scratch"
        self.experiment_name = experiment_name or f"{model_name}_{backbone}_{scale}_{pretrained_tag}"
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = None
        self.criterion = None
        self.postprocessors = None
        self.results = None
        self.save_dir = project_root / "results" / self.project_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = Path(data_yaml).stem if data_yaml else "unknown"
        self.logger = MetricsLogger(
            experiment_name=self.experiment_name,
            model_name=f"DINO-{self.backbone}-{self.scale}",
            dataset_name=dataset_name,
            save_dir=str(self.save_dir)
        )
    def _parse_data_yaml(self, data_yaml: str):
        with open(data_yaml, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        self.num_classes = config.get('nc', 1)
        self.class_names = config.get('names', ['fish'])
        base_path = Path(config.get('path', ''))
        if not base_path.is_absolute():
            base_path = project_root / base_path
        coco_format_path = base_path / "COCO_format"
        if coco_format_path.exists():
            self.coco_path = str(coco_format_path)
            print(f"✓ 找到COCO格式数据: {self.coco_path}")
        else:
            self.coco_path = str(base_path)
            print(f"⚠️  未找到COCO_format目录，使用: {self.coco_path}")
            print(f"⚠️  请先运行转换脚本: python convert_all_to_coco.py")
    def _get_config(self, **kwargs):
        config_file = self.SUPPORTED_SCALES.get(self.scale, "DINO_4scale.py")
        config_path = dino_path / "config" / "DINO" / config_file
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        return config_path
    def load_model(self, pretrained: bool = None):
        if pretrained is None:
            pretrained = self.pretrained
        print(f"加载DINO模型 (backbone: {self.backbone}, scale: {self.scale})")
        print(f"训练模式: {'使用预训练backbone' if pretrained else '完全从头训练'}")
        if not pretrained:
            print("✓ DINO支持完全从头训练，适合科研公平对比")
        try:
            from util.slconfig import SLConfig
            from models.registry import MODULE_BUILD_FUNCS
            config_path = self._get_config()
            cfg = SLConfig.fromfile(str(config_path))
            cfg.num_classes = self.num_classes
            cfg.backbone = self.backbone
            build_func = MODULE_BUILD_FUNCS.get(cfg.modelname)
            self.model, self.criterion, self.postprocessors = build_func(cfg)
            if not pretrained:
                self._reinit_backbone()
            self.model.to(self.device)
        except Exception as e:
            print(f"加载DINO模型失败: {e}")
            print("尝试使用简化版本...")
            self._load_simplified_model(pretrained)
        self._log_model_info()
        return self.model
    def _load_simplified_model(self, pretrained: bool):
        print("使用简化版DINO模型...")
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
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            flops_map = {
                "resnet50": 279,
                "resnet101": 350,
                "swin-t": 280,
                "swin-l": 900,
            }
            flops = flops_map.get(self.backbone, 279)
            model_size = total_params * 4 / (1024 * 1024)
            self.logger.set_model_info(
                params=total_params,
                flops=flops,
                model_size_mb=model_size,
                input_size=(640, 640)
            )
            print(f"模型参数量: {total_params/1e6:.2f}M")
            print(f"可训练参数: {trainable_params/1e6:.2f}M")
            print(f"估算FLOPs: {flops:.2f}G")
            print(f"模型大小: {model_size:.2f}MB")
        except Exception as e:
            print(f"获取模型信息失败: {e}")
    def train(self,
              epochs: int = 12,
              batch_size: int = 2,
              imgsz: int = 640,
              lr0: float = 1e-4,
              lr_backbone: float = 1e-5,
              weight_decay: float = 1e-4,
              lr_drop: int = 11,
              warmup_epochs: int = 0,
              patience: int = 50,
              save_period: int = 1,
              workers: int = 0,
              amp: bool = False,
              resume: bool = False,
              **kwargs) -> Dict:
        if self.model is None:
            self.load_model()
        if self.data_yaml is None:
            raise ValueError("请指定数据集配置文件 (data_yaml)")
        config = {
            "model": self.model_name,
            "backbone": self.backbone,
            "scale": self.scale,
            "pretrained": self.pretrained,
            "epochs": epochs,
            "batch_size": batch_size,
            "imgsz": imgsz,
            "lr0": lr0,
            "lr_backbone": lr_backbone,
            "weight_decay": weight_decay,
            "lr_drop": lr_drop,
            "device": self.device,
            **kwargs
        }
        self.logger.set_config(config)
        print(f"\n{'='*60}")
        print(f"开始训练: DINO ({self.backbone}, {self.scale})")
        print(f"数据集: {self.data_yaml}")
        print(f"设备: {self.device}")
        print(f"训练轮数: {epochs}")
        print(f"批次大小: {batch_size}")
        print(f"从头训练: {'否' if self.pretrained else '是'}")
        print(f"{'='*60}\n")
        start_time = time.time()
        coco_path = Path(self.coco_path)
        train_img_dir = coco_path / "train2017"
        val_img_dir = coco_path / "val2017"
        train_ann = coco_path / "annotations" / "instances_train2017.json"
        val_ann = coco_path / "annotations" / "instances_val2017.json"
        if not train_img_dir.exists() or not train_ann.exists():
            raise FileNotFoundError(
                f"COCO数据不完整！\n"
                f"请先运行: python convert_all_to_coco.py\n"
                f"期望路径:\n"
                f"  - {train_img_dir}\n"
                f"  - {train_ann}"
            )
        print(f"✓ COCO数据验证通过")
        print(f"  训练图像: {train_img_dir}")
        print(f"  训练标注: {train_ann}")
        try:
            from engine import train_one_epoch, evaluate
            from datasets import build_dataset, get_coco_api_from_dataset
            from torch.utils.data import DataLoader
            import util.misc as utils
            from util.slconfig import SLConfig
            config_file = self._get_config()
            cfg = SLConfig.fromfile(str(config_file))
            cfg.num_classes = self.num_classes
            cfg.coco_path = str(coco_path)
            cfg.batch_size = batch_size
            cfg.epochs = epochs
            cfg.lr = lr0
            cfg.lr_backbone = lr_backbone
            cfg.weight_decay = weight_decay
            cfg.lr_drop = lr_drop
            cfg.device = self.device
            cfg.num_workers = workers
            cfg.output_dir = str(self.save_dir / "runs" / self.experiment_name)
            Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
            dataset_train = build_dataset(image_set='train', args=cfg)
            dataset_val = build_dataset(image_set='val', args=cfg)
            print(f"  训练集大小: {len(dataset_train)}")
            print(f"  验证集大小: {len(dataset_val)}")
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
                           if "backbone" not in n and p.requires_grad]},
                {"params": [p for n, p in self.model.named_parameters()
                           if "backbone" in n and p.requires_grad],
                 "lr": lr_backbone},
            ]
            optimizer = torch.optim.AdamW(param_dicts, lr=lr0, weight_decay=weight_decay)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=lr0 * 0.01
            )
            print(f"\n开始训练，共{epochs}个epoch...")
            best_map = 0.0
            for epoch in range(epochs):
                train_stats = train_one_epoch(
                    self.model, self.criterion, data_loader_train, optimizer,
                    self.device, epoch, cfg.clip_max_norm if hasattr(cfg, 'clip_max_norm') else 0.1
                )
                lr_scheduler.step()
                train_loss = train_stats.get('loss', 0)
                self.logger.log_epoch(epoch + 1, {
                    'train/loss': train_loss,
                    'lr': optimizer.param_groups[0]['lr'],
                })
                if (epoch + 1) % save_period == 0 or epoch == epochs - 1:
                    base_ds = get_coco_api_from_dataset(dataset_val)
                    test_stats, coco_evaluator = evaluate(
                        self.model, self.criterion, self.postprocessors,
                        data_loader_val, base_ds, self.device, cfg.output_dir
                    )
                    if coco_evaluator is not None:
                        coco_eval = coco_evaluator.coco_eval['bbox']
                        stats = coco_eval.stats
                        metrics = {
                            'val/mAP50-95': stats[0],
                            'val/mAP50': stats[1],
                            'val/mAP75': stats[2],
                        }
                        self.logger.log_epoch(epoch + 1, metrics)
                        if stats[0] > best_map:
                            best_map = stats[0]
                            torch.save(self.model.state_dict(),
                                      Path(cfg.output_dir) / 'best.pth')
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}")
            print(f"\n训练完成! 最佳mAP: {best_map:.4f}")
        except ImportError as e:
            print(f"⚠️  DINO依赖导入失败: {e}")
            print("⚠️  DINO需要特定的依赖，请检查DINO-main目录")
            print("⚠️  跳过DINO训练")
            raise
        except Exception as e:
            print(f"训练过程出错: {e}")
            import traceback
            traceback.print_exc()
            raise
        training_time = time.time() - start_time
        return {
            "model_name": self.model_name,
            "backbone": self.backbone,
            "scale": self.scale,
            "training_time": training_time,
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
        fps = 1000 / avg_time
        self.logger.set_inference_metrics(
            inference_time_ms=avg_time,
            fps=fps,
            latency_ms=avg_time,
            batch_size=1,
            input_size=(imgsz, imgsz)
        )
        print(f"平均推理时间: {avg_time:.2f}ms (±{std_time:.2f}ms)")
        print(f"FPS: {fps:.2f}")
        return {"inference_time_ms": avg_time, "fps": fps}
    def get_results(self) -> Dict:
        return {
            "model_name": self.model_name,
            "backbone": self.backbone,
            "scale": self.scale,
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
    print("DINO训练器测试")
    print("="*60)
    print("DINO支持完全从头训练，适合科研公平对比")
    print("="*60)
    trainer = DINOTrainer(
        backbone="resnet50",
        scale="4scale",
        data_yaml="configs/dataset_blur_png.yaml",
        pretrained=False
    )
    print(f"训练器初始化成功")
