
import os
import sys
import time
import warnings
import torch
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np
warnings.filterwarnings('ignore', message='.*grid_sampler_2d_backward_cuda.*')
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils.metrics_logger import MetricsLogger, ExperimentTracker
class RTDETRTrainer:
    SUPPORTED_MODELS = {
        "rtdetr-s": "rtdetr-l.pt",
        "rtdetr-m": "rtdetr-l.pt",
        "rtdetr-l": "rtdetr-l.pt",
        "rtdetr-x": "rtdetr-x.pt",
        "rtdetr-r18": "rtdetr-l.pt",
        "rtdetr-r34": "rtdetr-l.pt",
        "rtdetr-r50": "rtdetr-l.pt",
        "rtdetr-r50m": "rtdetr-l.pt",
        "rtdetr-r101": "rtdetr-l.pt",
    }
    def __init__(self,
                 model_name: str = "rtdetr-l",
                 data_yaml: str = None,
                 project_name: str = "fish_detection",
                 experiment_name: str = None,
                 device: str = "auto",
                 pretrained: bool = True):
        self.model_name = model_name.lower()
        self.data_yaml = data_yaml
        self.project_name = project_name
        self.pretrained = pretrained
        if not pretrained:
            pretrained_tag = "partial_from_scratch"
        else:
            pretrained_tag = "pretrained"
        self.experiment_name = experiment_name or f"{model_name}_{pretrained_tag}"
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = None
        self.results = None
        self._partial_from_scratch = False
        self.save_dir = project_root / "results" / self.project_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = Path(data_yaml).stem if data_yaml else "unknown"
        self.logger = MetricsLogger(
            experiment_name=self.experiment_name,
            model_name=self.model_name.upper(),
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
        if pretrained:
            model_name_map = {
                "rtdetr-s": "rtdetr-l.pt",
                "rtdetr-m": "rtdetr-l.pt",
                "rtdetr-l": "rtdetr-l.pt",
                "rtdetr-x": "rtdetr-x.pt",
            }
            if self.model_name in model_name_map:
                weights = model_name_map[self.model_name]
            else:
                weights = f"{self.model_name}.pt"
            print(f"加载RT-DETR模型 (使用预训练权重): {weights}")
            print(f"  ultralytics会自动下载预训练权重（如果本地不存在）")
            print(f"  如果下载失败，请检查网络连接或使用可用的模型: rtdetr-l, rtdetr-x")
            try:
                self.model = YOLO(weights)
            except Exception as e:
                if weights != "rtdetr-l.pt":
                    print(f"⚠️  模型 {weights} 加载失败: {e}")
                    print(f"  尝试使用替代模型: rtdetr-l.pt")
                    try:
                        self.model = YOLO("rtdetr-l.pt")
                        print(f"✓ 成功加载替代模型: rtdetr-l.pt")
                    except Exception as e2:
                        raise RuntimeError(
                            f"无法加载RT-DETR模型。\n"
                            f"原始错误: {e}\n"
                            f"替代模型错误: {e2}\n"
                            f"请检查:\n"
                            f"1. 网络连接（需要下载预训练权重）\n"
                            f"2. ultralytics版本: pip install --upgrade ultralytics\n"
                            f"3. 或手动下载模型文件"
                        ) from e2
                else:
                    raise
        else:
            raise ValueError(
                "RT-DETR在ultralytics中不支持完全从头训练（没有YAML配置文件）。\n"
                "科研公平对比建议：\n"
                "1. 所有模型都使用预训练权重进行对比（推荐）\n"
                "   python run_experiments.py --model yolo11s rtdetr-s --pretrained\n"
                "2. 或者跳过RT-DETR，只对比支持完全从头训练的模型\n"
                "   python run_experiments.py --model yolo11s yolo11n --from-scratch\n"
                "3. 如需RT-DETR完全从头训练，需要使用官方PaddleDetection实现"
            )
        self._log_model_info()
        return self.model
    def _log_model_info(self):
        if self.model is None:
            return
        try:
            params = sum(p.numel() for p in self.model.model.parameters())
            flops_map = {
                "rtdetr-s": 60,
                "rtdetr-m": 100,
                "rtdetr-l": 136,
                "rtdetr-x": 259,
                "rtdetr-r18": 60,
                "rtdetr-r34": 92,
                "rtdetr-r50": 136,
                "rtdetr-r101": 259,
            }
            flops = flops_map.get(self.model_name, params * 2 / 1e9)
            model_size = params * 4 / (1024 * 1024)
            self.logger.set_model_info(
                params=params,
                flops=flops,
                model_size_mb=model_size,
                input_size=(640, 640)
            )
            print(f"模型参数量: {params/1e6:.2f}M")
            print(f"FLOPs: {flops:.2f}G")
            print(f"模型大小: {model_size:.2f}MB")
        except Exception as e:
            print(f"获取模型信息失败: {e}")
    def train(self,
              epochs: int = 100,
              batch_size: int = 16,
              imgsz: int = 640,
              lr0: float = 0.0001,
              lrf: float = 0.01,
              weight_decay: float = 0.0001,
              warmup_epochs: float = 3.0,
              patience: int = 50,
              save_period: int = -1,
              workers: int = 0,
              amp: bool = True,
              resume: bool = False,
              optimizer: str = "AdamW",
              plots: bool = False,
              **kwargs) -> Dict:
        if self.model is None:
            self.load_model()
        if self.data_yaml is None:
            raise ValueError("请指定数据集配置文件 (data_yaml)")
        config = {
            "model": self.model_name,
            "pretrained": self.pretrained,
            "partial_from_scratch": getattr(self, '_partial_from_scratch', False),
            "epochs": epochs,
            "batch_size": batch_size,
            "imgsz": imgsz,
            "lr0": lr0,
            "lrf": lrf,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "patience": patience,
            "optimizer": optimizer,
            "device": self.device,
            "amp": amp,
            **kwargs
        }
        self.logger.set_config(config)
        print(f"\n{'='*60}")
        print(f"开始训练: RT-DETR ({self.model_name})")
        print(f"数据集: {self.data_yaml}")
        print(f"设备: {self.device}")
        print(f"训练轮数: {epochs}")
        print(f"批次大小: {batch_size}")
        print(f"优化器: {optimizer}")
        print(f"{'='*60}\n")
        start_time = time.time()
        self.results = self.model.train(
            data=self.data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            lr0=lr0,
            lrf=lrf,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            patience=patience,
            save_period=save_period,
            workers=workers,
            device=self.device,
            project=str(self.save_dir / "runs"),
            name=self.experiment_name,
            exist_ok=True,
            amp=amp,
            resume=resume,
            optimizer=optimizer,
            verbose=True,
            plots=plots,
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
    def validate(self, data_yaml: str = None, conf: float = 0.25, iou: float = 0.6) -> Dict:
        if self.model is None:
            raise ValueError("请先加载或训练模型")
        data = data_yaml or self.data_yaml
        print(f"\n开始验证...")
        metrics = self.model.val(
            data=data,
            conf=conf,
            iou=iou,
            device=self.device
        )
        results = {
            "mAP@0.5": metrics.box.map50,
            "mAP@0.5:0.95": metrics.box.map,
            "precision": metrics.box.mp,
            "recall": metrics.box.mr,
        }
        print(f"验证结果:")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
        return results
    def benchmark_inference(self,
                           num_runs: int = 100,
                           warmup_runs: int = 10,
                           imgsz: int = 640) -> Dict:
        if self.model is None:
            raise ValueError("请先加载或训练模型")
        print(f"\n测试RT-DETR推理性能...")
        print(f"注意: RT-DETR是端到端检测器，无需NMS")
        dummy_input = torch.rand(1, 3, imgsz, imgsz).to(self.device)
        print(f"预热 {warmup_runs} 次...")
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = self.model.predict(dummy_input, verbose=False)
        if self.device == "cuda":
            torch.cuda.synchronize()
        print(f"测试 {num_runs} 次...")
        times = []
        for _ in range(num_runs):
            if self.device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = self.model.predict(dummy_input, verbose=False)
            if self.device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        times = np.array(times)
        inference_time = np.mean(times)
        fps = 1000 / inference_time
        self.logger.log_inference_metrics(
            inference_time_ms=inference_time,
            preprocess_time_ms=0,
            postprocess_time_ms=0,
            nms_time_ms=0,
            fps=fps,
            batch_size=1,
            device=self.device,
            precision="fp32"
        )
        results = {
            "inference_time_ms": inference_time,
            "inference_time_std_ms": np.std(times),
            "end_to_end_latency_ms": inference_time,
            "fps": fps,
            "min_time_ms": np.min(times),
            "max_time_ms": np.max(times),
            "device": self.device,
            "nms_free": True
        }
        print(f"\nRT-DETR推理性能:")
        print(f"  端到端延迟: {inference_time:.2f} ± {np.std(times):.2f} ms")
        print(f"  FPS: {fps:.1f}")
        print(f"  NMS-Free: Yes")
        return results
    def export(self, format: str = "onnx", **kwargs) -> str:
        if self.model is None:
            raise ValueError("请先加载或训练模型")
        print(f"\n导出RT-DETR模型为 {format} 格式...")
        path = self.model.export(format=format, **kwargs)
        print(f"模型已导出到: {path}")
        return path
    def get_results(self) -> Dict:
        return {
            "model_name": self.model_name,
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
def train_rtdetr(model_name: str = "rtdetr-l",
                 data_yaml: str = None,
                 epochs: int = 100,
                 batch_size: int = 16,
                 imgsz: int = 640,
                 device: str = "auto",
                 pretrained: bool = True,
                 **kwargs) -> Dict:
    trainer = RTDETRTrainer(
        model_name=model_name,
        data_yaml=data_yaml,
        device=device,
        pretrained=pretrained
    )
    trainer.load_model()
    results = trainer.train(
        epochs=epochs,
        batch_size=batch_size,
        imgsz=imgsz,
        **kwargs
    )
    trainer.benchmark_inference()
    trainer.save_results()
    return results
if __name__ == "__main__":
    print("RT-DETR训练器测试")
    try:
        from ultralytics import RTDETR
        print("ultralytics RTDETR 已安装")
        trainer = RTDETRTrainer(
            model_name="rtdetr-l",
            data_yaml="configs/dataset_blur_png.yaml"
        )
        print(f"训练器初始化成功")
    except ImportError:
        print("请安装 ultralytics: pip install ultralytics")
