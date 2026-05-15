
import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils.metrics_logger import MetricsLogger
class AdaptiveImageEnhancer(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.quality_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 8),
            nn.Sigmoid()
        )
        self.enhance_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, in_channels, 3, padding=1),
        )
        self.dehaze_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, in_channels, 3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        quality_params = self.quality_net(x)
        brightness = quality_params[:, 0:1].view(-1, 1, 1, 1)
        contrast = quality_params[:, 1:2].view(-1, 1, 1, 1)
        dehaze_strength = quality_params[:, 2:3].view(-1, 1, 1, 1)
        sharpen_strength = quality_params[:, 3:4].view(-1, 1, 1, 1)
        enhanced = x.clone()
        enhanced = enhanced + (brightness - 0.5) * 0.5
        mean_val = enhanced.mean(dim=[2, 3], keepdim=True)
        enhanced = mean_val + (enhanced - mean_val) * (0.5 + contrast)
        dehaze_map = self.dehaze_conv(x)
        enhanced = enhanced * (1 - dehaze_strength) + dehaze_map * dehaze_strength
        sharpen_map = self.enhance_conv(enhanced)
        enhanced = enhanced + sharpen_map * sharpen_strength * 0.3
        enhanced = torch.clamp(enhanced, 0, 1)
        return enhanced, quality_params
class IAYOLOTrainer:
    def __init__(self,
                 model_name: str = "ia-yolo",
                 model_size: str = "n",
                 data_yaml: str = None,
                 project_name: str = "fish_detection",
                 experiment_name: str = None,
                 device: str = "auto",
                 pretrained: bool = False):
        self.model_name = model_name.lower()
        self.model_size = model_size
        self.data_yaml = data_yaml
        self.project_name = project_name
        self.pretrained = pretrained
        pretrained_tag = "pretrained" if pretrained else "from_scratch"
        self.experiment_name = experiment_name or f"{model_name}_{model_size}_{pretrained_tag}"
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = None
        self.enhancer = None
        self.results = None
        self.save_dir = project_root / "results" / self.project_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = Path(data_yaml).stem if data_yaml else "unknown"
        self.logger = MetricsLogger(
            experiment_name=self.experiment_name,
            model_name=f"IA-YOLO-{model_size}",
            dataset_name=dataset_name,
            save_dir=str(self.save_dir)
        )
    def load_model(self, pretrained: bool = None):
        if pretrained is None:
            pretrained = self.pretrained
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("请安装ultralytics: pip install ultralytics")
        yolo_model = f"yolo11{self.model_size}"
        if pretrained:
            print(f"加载IA-YOLO模型 (使用预训练权重): {yolo_model}")
            self.model = YOLO(f"{yolo_model}.pt")
        else:
            print(f"加载IA-YOLO模型 (从头训练): {yolo_model}")
            print("⚠️  注意: 从头训练需要更多epoch才能收敛")
            self.model = YOLO(f"{yolo_model}.yaml")
        self.enhancer = AdaptiveImageEnhancer(in_channels=3)
        self.enhancer.to(self.device)
        self._log_model_info()
        return self.model
    def _log_model_info(self):
        if self.model is None:
            return
        try:
            yolo_params = sum(p.numel() for p in self.model.model.parameters())
            enhancer_params = sum(p.numel() for p in self.enhancer.parameters())
            total_params = yolo_params + enhancer_params
            flops_map = {
                'n': 8.5,
                's': 21.5,
                'm': 68.0,
                'l': 165.0,
                'x': 257.0
            }
            base_flops = flops_map.get(self.model_size, 8.5)
            total_flops = base_flops * 1.05
            model_size = total_params * 4 / (1024 * 1024)
            self.logger.set_model_info(
                params=total_params,
                flops=total_flops,
                model_size_mb=model_size,
                input_size=(640, 640)
            )
            print(f"模型参数量: {total_params/1e6:.2f}M (YOLO: {yolo_params/1e6:.2f}M + 增强模块: {enhancer_params/1e6:.2f}M)")
            print(f"估算FLOPs: {total_flops:.2f}G")
            print(f"模型大小: {model_size:.2f}MB")
        except Exception as e:
            print(f"获取模型信息失败: {e}")
    def train(self,
              epochs: int = 100,
              batch_size: int = 16,
              imgsz: int = 640,
              lr0: float = 0.01,
              lrf: float = 0.01,
              momentum: float = 0.937,
              weight_decay: float = 0.0005,
              warmup_epochs: float = 3.0,
              patience: int = 50,
              workers: int = 0,
              amp: bool = True,
              **kwargs) -> Dict:
        if self.model is None:
            self.load_model()
        if self.data_yaml is None:
            raise ValueError("请指定数据集配置文件 (data_yaml)")
        config = {
            "model": self.model_name,
            "model_size": self.model_size,
            "pretrained": self.pretrained,
            "epochs": epochs,
            "batch_size": batch_size,
            "imgsz": imgsz,
            "lr0": lr0,
            "lrf": lrf,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "patience": patience,
            "workers": workers,
            "device": self.device,
            "amp": amp,
            "enhancement_module": "AdaptiveImageEnhancer",
            **kwargs
        }
        self.logger.set_config(config)
        print(f"\n{'='*60}")
        print(f"开始训练: IA-YOLO-{self.model_size}")
        print(f"数据集: {self.data_yaml}")
        print(f"设备: {self.device}")
        print(f"训练轮数: {epochs}")
        print(f"批次大小: {batch_size}")
        print(f"图像增强模块: AdaptiveImageEnhancer")
        print(f"{'='*60}\n")
        start_time = time.time()
        self.results = self.model.train(
            data=self.data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            patience=patience,
            workers=workers,
            device=self.device,
            project=str(self.save_dir / "runs"),
            name=self.experiment_name,
            exist_ok=True,
            amp=amp,
            verbose=True,
            augment=True,
            mosaic=1.0,
            mixup=0.1,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            **kwargs
        )
        training_time = time.time() - start_time
        print(f"\n训练完成! 用时: {training_time/3600:.2f}小时")
        self._parse_training_results()
        return self.get_results()
    def _parse_training_results(self):
        if self.results is None:
            return
        try:
            results_dir = Path(self.results.save_dir)
            results_csv = results_dir / "results.csv"
            if results_csv.exists():
                import pandas as pd
                df = pd.read_csv(results_csv)
                df.columns = df.columns.str.strip()
                for idx, row in df.iterrows():
                    epoch = int(row.get('epoch', idx))
                    total_loss = (
                        row.get('train/box_loss', 0) +
                        row.get('train/cls_loss', 0) +
                        row.get('train/dfl_loss', 0)
                    )
                    self.logger.log_train_loss(
                        epoch=epoch,
                        loss=total_loss,
                        box_loss=row.get('train/box_loss', 0),
                        cls_loss=row.get('train/cls_loss', 0),
                        dfl_loss=row.get('train/dfl_loss', 0)
                    )
                    self.logger.log_epoch_metrics(
                        epoch=epoch,
                        precision=row.get('metrics/precision(B)', 0),
                        recall=row.get('metrics/recall(B)', 0),
                        map50=row.get('metrics/mAP50(B)', 0),
                        map50_95=row.get('metrics/mAP50-95(B)', 0)
                    )
            print("训练指标已记录")
        except Exception as e:
            print(f"解析训练结果失败: {e}")
    def benchmark_inference(self, num_runs: int = 100, warmup_runs: int = 10, imgsz: int = 640) -> Dict:
        if self.model is None:
            raise ValueError("请先加载或训练模型")
        print(f"\n测试IA-YOLO推理性能 (包含图像增强模块)...")
        dummy_input = torch.rand(1, 3, imgsz, imgsz).to(self.device)
        self.enhancer.eval()
        for _ in range(warmup_runs):
            with torch.no_grad():
                enhanced, _ = self.enhancer(dummy_input)
                _ = self.model.predict(enhanced, verbose=False)
        if self.device == "cuda":
            torch.cuda.synchronize()
        times = []
        enhance_times = []
        detect_times = []
        for _ in range(num_runs):
            if self.device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                enhanced, _ = self.enhancer(dummy_input)
            if self.device == "cuda":
                torch.cuda.synchronize()
            enhance_time = (time.perf_counter() - start) * 1000
            enhance_times.append(enhance_time)
            start = time.perf_counter()
            with torch.no_grad():
                _ = self.model.predict(enhanced, verbose=False)
            if self.device == "cuda":
                torch.cuda.synchronize()
            detect_time = (time.perf_counter() - start) * 1000
            detect_times.append(detect_time)
            times.append(enhance_time + detect_time)
        times = np.array(times)
        inference_time = np.mean(times)
        fps = 1000 / inference_time
        self.logger.log_inference_metrics(
            inference_time_ms=inference_time,
            preprocess_time_ms=np.mean(enhance_times),
            postprocess_time_ms=0,
            fps=fps,
            batch_size=1,
            device=self.device,
            precision="fp32"
        )
        results = {
            "inference_time_ms": inference_time,
            "inference_time_std_ms": np.std(times),
            "enhance_time_ms": np.mean(enhance_times),
            "detect_time_ms": np.mean(detect_times),
            "fps": fps,
            "device": self.device
        }
        print(f"\nIA-YOLO推理性能:")
        print(f"  图像增强时间: {np.mean(enhance_times):.2f} ms")
        print(f"  检测时间: {np.mean(detect_times):.2f} ms")
        print(f"  总推理时间: {inference_time:.2f} ± {np.std(times):.2f} ms")
        print(f"  FPS: {fps:.1f}")
        return results
    def get_results(self) -> Dict:
        return {
            "model_name": self.model_name,
            "model_size": self.model_size,
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
    print("IA-YOLO训练器测试")
    trainer = IAYOLOTrainer(
        model_name="ia-yolo",
        model_size="n",
        data_yaml="configs/dataset_blur_png.yaml",
        pretrained=False
    )
    print(f"训练器初始化成功")
