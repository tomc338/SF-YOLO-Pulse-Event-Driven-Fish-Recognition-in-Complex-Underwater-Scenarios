
import os
import sys
import time
import torch
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
ultralytics_src = project_root / "ultralytics"
if ultralytics_src.exists() and (ultralytics_src / "ultralytics").exists():
    if str(ultralytics_src) not in sys.path:
        sys.path.insert(0, str(ultralytics_src))
from utils.metrics_logger import MetricsLogger, ExperimentTracker
class YOLOTrainer:
    SUPPORTED_MODELS = {
        "yolo26n": "yolo26n.pt",
        "yolo26s": "yolo26s.pt",
        "yolo26m": "yolo26m.pt",
        "yolo26l": "yolo26l.pt",
        "yolo26x": "yolo26x.pt",
        "yolo11n": "yolo11n.pt",
        "yolo11s": "yolo11s.pt",
        "yolo11m": "yolo11m.pt",
        "yolo11l": "yolo11l.pt",
        "yolo11x": "yolo11x.pt",
        "yolov10n": "yolov10n.pt",
        "yolov10s": "yolov10s.pt",
        "yolov10m": "yolov10m.pt",
        "yolov10l": "yolov10l.pt",
        "yolov10x": "yolov10x.pt",
        "yolov8n": "yolov8n.pt",
        "yolov8s": "yolov8s.pt",
        "yolov8m": "yolov8m.pt",
        "yolov8l": "yolov8l.pt",
        "yolov8x": "yolov8x.pt",
    }
    def __init__(self,
                 model_name: str = "yolo11n",
                 data_yaml: str = None,
                 project_name: str = "fish_detection",
                 experiment_name: str = None,
                 device: str = "auto",
                 pretrained: bool = True):
        self.model_name = model_name.lower()
        self.data_yaml = data_yaml
        self.project_name = project_name
        self.pretrained = pretrained
        pretrained_tag = "pretrained" if pretrained else "from_scratch"
        self.experiment_name = experiment_name or f"{model_name}_{pretrained_tag}"
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = None
        self.results = None
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
        except ImportError as e:
            import_error_msg = (
                f"无法导入 ultralytics: {e}\n"
                f"请尝试以下方法之一：\n"
                f"1. 安装 ultralytics: pip install ultralytics\n"
                f"2. 或者确保 ultralytics 源码在项目根目录的 ultralytics/ 文件夹中"
            )
            raise ImportError(import_error_msg)
        if pretrained:
            if self.model_name in self.SUPPORTED_MODELS:
                weights = self.SUPPORTED_MODELS[self.model_name]
            else:
                weights = f"{self.model_name}.pt"
            import os
            if os.path.exists(weights) or weights.startswith('yolo') or weights.startswith('rtdetr'):
                print(f"加载模型 (使用预训练权重): {weights}")
                self.model = YOLO(weights)
            else:
                yaml_file = self._get_yaml_path()
                print(f"⚠️  预训练权重文件不存在: {weights}")
                print(f"自动切换到从头训练: {yaml_file}")
                print("⚠️  注意: 从头训练需要更多epoch才能收敛")
                self.model = YOLO(yaml_file)
        else:
            yaml_file = self._get_yaml_path()
            print(f"加载模型 (从头训练，不使用预训练权重): {yaml_file}")
            print("⚠️  注意: 从头训练需要更多epoch才能收敛")
            self.model = YOLO(yaml_file)
        self._log_model_info()
        return self.model
    def _get_yaml_path(self):
        yaml_file = f"{self.model_name}.yaml"
        possible_paths = [
            yaml_file,
            f"ultralytics/ultralytics/cfg/models/26/{yaml_file}",
            f"ultralytics/ultralytics/cfg/models/v8/{yaml_file}",
            f"ultralytics/ultralytics/cfg/models/v10/{yaml_file}",
            f"ultralytics/ultralytics/cfg/models/v11/{yaml_file}",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return yaml_file
    def _log_model_info(self):
        if self.model is None:
            return
        try:
            params = sum(p.numel() for p in self.model.model.parameters())
            flops = params * 2 / 1e9
            model_size = params * 4 / (1024 * 1024)
            self.logger.set_model_info(
                params=params,
                flops=flops,
                model_size_mb=model_size,
                input_size=(640, 640)
            )
            print(f"模型参数量: {params/1e6:.2f}M")
            print(f"估算FLOPs: {flops:.2f}G")
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
              save_period: int = -1,
              workers: int = 0,
              amp: bool = True,
              resume: bool = False,
              ema: bool = True,
              iou_type: str = None,
              **kwargs) -> Dict:
        if self.model is None:
            self.load_model()
        if self.data_yaml is None:
            raise ValueError("请指定数据集配置文件 (data_yaml)")
        config = {
            "model": self.model_name,
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
            "ema": ema,
            "device": self.device,
            "amp": amp,
            **kwargs
        }
        self.logger.set_config(config)
        print(f"\n{'='*60}")
        print(f"开始训练: {self.model_name}")
        print(f"数据集: {self.data_yaml}")
        print(f"设备: {self.device}")
        print(f"训练轮数: {epochs}")
        print(f"批次大小: {batch_size}")
        print(f"{'='*60}\n")
        if hasattr(self.model, 'args') and iou_type is not None:
            if isinstance(self.model.args, dict):
                self.model.args['iou_type'] = iou_type
            else:
                setattr(self.model.args, 'iou_type', iou_type)
            print(f"使用IoU损失类型: {iou_type}")
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
            save_period=save_period,
            workers=workers,
            device=self.device,
            project=str(self.save_dir / "runs"),
            name=self.experiment_name,
            exist_ok=True,
            amp=amp,
            resume=resume,
            verbose=True,
            **kwargs
        )
        training_time = time.time() - start_time
        print(f"\n训练完成! 用时: {training_time/3600:.2f}小时")
        self._parse_training_results()
        self._save_best_model()
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
                    self.logger.log_train_loss(
                        epoch=epoch,
                        loss=row.get('train/box_loss', 0) + row.get('train/cls_loss', 0) + row.get('train/dfl_loss', 0),
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
        print(f"\n测试推理性能...")
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
            fps=fps,
            batch_size=1,
            device=self.device,
            precision="fp32"
        )
        results = {
            "inference_time_ms": inference_time,
            "inference_time_std_ms": np.std(times),
            "fps": fps,
            "min_time_ms": np.min(times),
            "max_time_ms": np.max(times),
            "device": self.device
        }
        print(f"\n推理性能:")
        print(f"  平均推理时间: {inference_time:.2f} ± {np.std(times):.2f} ms")
        print(f"  FPS: {fps:.1f}")
        print(f"  最小时间: {np.min(times):.2f} ms")
        print(f"  最大时间: {np.max(times):.2f} ms")
        return results
    def export(self, format: str = "onnx", **kwargs) -> str:
        if self.model is None:
            raise ValueError("请先加载或训练模型")
        print(f"\n导出模型为 {format} 格式...")
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
    def _save_best_model(self):
        if self.results is None:
            return
        try:
            results_dir = Path(self.results.save_dir)
            best_model_path = results_dir / "weights" / "best.pt"
            if not best_model_path.exists():
                print(f"⚠️  未找到最优模型: {best_model_path}")
                return
            yolo_best_dir = project_root / "yolo_best"
            yolo_best_dir.mkdir(parents=True, exist_ok=True)
            if self.data_yaml:
                dataset_name = Path(self.data_yaml).stem
                dataset_name = dataset_name.replace("dataset_", "").replace("_", "-")
            else:
                dataset_name = "unknown"
            save_filename = f"{self.model_name}_{dataset_name}_best.pt"
            save_path = yolo_best_dir / save_filename
            if save_path.exists():
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_filename = f"{self.model_name}_{dataset_name}_{timestamp}_best.pt"
                save_path = yolo_best_dir / save_filename
            import shutil
            shutil.copy2(best_model_path, save_path)
            print(f"\n✓ 最优模型已保存到: {save_path}")
            print(f"  源文件: {best_model_path}")
        except Exception as e:
            print(f"⚠️  保存最优模型时出错: {e}")
    def save_results(self):
        excel_path, json_path = self.logger.save_all()
        print(f"\n结果已保存:")
        print(f"  Excel: {excel_path}")
        print(f"  JSON: {json_path}")
        return excel_path, json_path
def train_yolo(model_name: str = "yolo11n",
               data_yaml: str = None,
               epochs: int = 100,
               batch_size: int = 16,
               imgsz: int = 640,
               device: str = "auto",
               pretrained: bool = True,
               **kwargs) -> Dict:
    trainer = YOLOTrainer(
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
    print("YOLO训练器测试")
    try:
        from ultralytics import YOLO
        print("ultralytics 已安装")
        trainer = YOLOTrainer(
            model_name="yolo11n",
            data_yaml="configs/dataset_blur_png.yaml"
        )
        print(f"训练器初始化成功")
        print(f"保存目录: {trainer.save_dir}")
    except ImportError:
        print("请安装 ultralytics: pip install ultralytics")
