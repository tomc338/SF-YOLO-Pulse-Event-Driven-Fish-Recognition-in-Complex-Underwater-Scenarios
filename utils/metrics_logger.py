
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
class MetricsLogger:
    def __init__(self,
                 experiment_name: str,
                 model_name: str,
                 dataset_name: str,
                 save_dir: str = "results"):
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = self.save_dir / f"{model_name}_{dataset_name}_{self.timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_metrics: List[Dict] = []
        self.train_losses: List[Dict] = []
        self.inference_metrics: Dict = {}
        self.model_info: Dict = {}
        self.pr_curve_data: Dict = {}
        self.config: Dict = {}
    def set_config(self, config: Dict):
        self.config = config
    def set_model_info(self,
                       params: int = 0,
                       flops: float = 0,
                       model_size_mb: float = 0,
                       input_size: tuple = (640, 640)):
        self.model_info = {
            "parameters": params,
            "flops_gflops": flops,
            "model_size_mb": model_size_mb,
            "input_size": input_size
        }
    def log_train_loss(self, epoch: int, loss: float,
                       box_loss: float = 0,
                       cls_loss: float = 0,
                       dfl_loss: float = 0,
                       **kwargs):
        loss_dict = {
            "epoch": epoch,
            "total_loss": loss,
            "box_loss": box_loss,
            "cls_loss": cls_loss,
            "dfl_loss": dfl_loss,
            **kwargs
        }
        self.train_losses.append(loss_dict)
    def log_epoch_metrics(self,
                          epoch: int,
                          precision: float = 0,
                          recall: float = 0,
                          map50: float = 0,
                          map50_95: float = 0,
                          f1: float = 0,
                          **kwargs):
        metrics_dict = {
            "epoch": epoch,
            "precision": precision,
            "recall": recall,
            "mAP@0.5": map50,
            "mAP@0.5:0.95": map50_95,
            "F1": f1 if f1 > 0 else 2 * precision * recall / (precision + recall + 1e-8),
            **kwargs
        }
        self.epoch_metrics.append(metrics_dict)
    def log_epoch(self, epoch: int, metrics: Dict):
        train_loss = metrics.get('train/loss', metrics.get('loss', 0))
        box_loss = metrics.get('train/loss_bbox', metrics.get('loss_bbox', 0))
        cls_loss = metrics.get('train/loss_ce', metrics.get('loss_ce', 0))
        giou_loss = metrics.get('train/loss_giou', metrics.get('loss_giou', 0))
        if train_loss > 0:
            self.log_train_loss(
                epoch=epoch,
                loss=train_loss,
                box_loss=box_loss,
                cls_loss=cls_loss,
                giou_loss=giou_loss
            )
        map50 = metrics.get('val/mAP50', metrics.get('mAP@0.5', 0))
        map50_95 = metrics.get('val/mAP50-95', metrics.get('mAP@0.5:0.95', 0))
        precision = metrics.get('val/precision', metrics.get('precision', 0))
        recall = metrics.get('val/recall', metrics.get('recall', 0))
        if map50 > 0 or map50_95 > 0:
            self.log_epoch_metrics(
                epoch=epoch,
                precision=precision,
                recall=recall,
                map50=map50,
                map50_95=map50_95
            )
    def log_inference_metrics(self,
                              inference_time_ms: float,
                              preprocess_time_ms: float = 0,
                              postprocess_time_ms: float = 0,
                              nms_time_ms: float = 0,
                              fps: float = 0,
                              batch_size: int = 1,
                              device: str = "cuda",
                              precision: str = "fp32"):
        total_latency = preprocess_time_ms + inference_time_ms + postprocess_time_ms
        self.inference_metrics = {
            "inference_time_ms": inference_time_ms,
            "preprocess_time_ms": preprocess_time_ms,
            "postprocess_time_ms": postprocess_time_ms,
            "nms_time_ms": nms_time_ms,
            "end_to_end_latency_ms": total_latency,
            "fps": fps if fps > 0 else 1000 / (inference_time_ms + 1e-8),
            "batch_size": batch_size,
            "device": device,
            "precision": precision
        }
    def set_inference_metrics(self,
                              inference_time_ms: float,
                              fps: float = 0,
                              latency_ms: float = 0,
                              batch_size: int = 1,
                              input_size: tuple = (640, 640),
                              device: str = "cuda",
                              **kwargs):
        self.inference_metrics = {
            "inference_time_ms": inference_time_ms,
            "end_to_end_latency_ms": latency_ms if latency_ms > 0 else inference_time_ms,
            "fps": fps if fps > 0 else 1000 / (inference_time_ms + 1e-8),
            "batch_size": batch_size,
            "input_size": input_size,
            "device": device,
            **kwargs
        }
    def log_pr_curve(self, precision_list: List[float], recall_list: List[float],
                     class_name: str = "fish"):
        self.pr_curve_data[class_name] = {
            "precision": precision_list,
            "recall": recall_list
        }
    def get_best_metrics(self) -> Dict:
        if not self.epoch_metrics:
            return {}
        df = pd.DataFrame(self.epoch_metrics)
        best_idx = df["mAP@0.5"].idxmax()
        best_metrics = df.iloc[best_idx].to_dict()
        best_metrics["best_epoch"] = int(best_metrics["epoch"])
        return best_metrics
    def save_to_excel(self, excel_path: Optional[str] = None):
        if excel_path is None:
            excel_path = self.exp_dir / f"{self.experiment_name}_metrics.xlsx"
        else:
            excel_path = Path(excel_path)
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            info_df = pd.DataFrame([{
                "实验名称": self.experiment_name,
                "模型": self.model_name,
                "数据集": self.dataset_name,
                "时间戳": self.timestamp,
                **self.model_info
            }])
            info_df.to_excel(writer, sheet_name="实验信息", index=False)
            if self.config:
                config_df = pd.DataFrame([self.config])
                config_df.to_excel(writer, sheet_name="训练配置", index=False)
            if self.train_losses:
                loss_df = pd.DataFrame(self.train_losses)
                loss_df.to_excel(writer, sheet_name="训练Loss", index=False)
            if self.epoch_metrics:
                metrics_df = pd.DataFrame(self.epoch_metrics)
                metrics_df.to_excel(writer, sheet_name="Epoch指标", index=False)
            if self.inference_metrics:
                inference_df = pd.DataFrame([self.inference_metrics])
                inference_df.to_excel(writer, sheet_name="推理性能", index=False)
            best_metrics = self.get_best_metrics()
            if best_metrics:
                best_df = pd.DataFrame([best_metrics])
                best_df.to_excel(writer, sheet_name="最佳指标", index=False)
        print(f"指标已保存到: {excel_path}")
        return excel_path
    def save_to_json(self, json_path: Optional[str] = None):
        if json_path is None:
            json_path = self.exp_dir / f"{self.experiment_name}_metrics.json"
        else:
            json_path = Path(json_path)
        data = {
            "experiment_name": self.experiment_name,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "config": self.config,
            "model_info": self.model_info,
            "train_losses": self.train_losses,
            "epoch_metrics": self.epoch_metrics,
            "inference_metrics": self.inference_metrics,
            "pr_curve_data": self.pr_curve_data,
            "best_metrics": self.get_best_metrics()
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"指标已保存到: {json_path}")
        return json_path
    def save_all(self):
        excel_path = self.save_to_excel()
        json_path = self.save_to_json()
        return excel_path, json_path
class ExperimentTracker:
    def __init__(self, project_name: str, save_dir: str = "results"):
        self.project_name = project_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.summary_file = self.save_dir / f"{project_name}_实验汇总.xlsx"
        self.experiments: List[Dict] = []
        if self.summary_file.exists():
            self._load_existing()
    def _load_existing(self):
        try:
            df = pd.read_excel(self.summary_file, sheet_name="实验汇总")
            self.experiments = df.to_dict('records')
        except Exception as e:
            print(f"加载现有数据失败: {e}")
            self.experiments = []
    def add_experiment(self, logger: MetricsLogger):
        best_metrics = logger.get_best_metrics()
        pretrained = logger.config.get("pretrained", True)
        partial_from_scratch = logger.config.get("partial_from_scratch", False)
        if partial_from_scratch:
            pretrained_str = "部分从头训练(RT-DETR)"
        elif pretrained:
            pretrained_str = "是"
        else:
            pretrained_str = "否(从头训练)"
        experiment_record = {
            "实验ID": len(self.experiments) + 1,
            "时间戳": logger.timestamp,
            "模型": logger.model_name,
            "数据集": logger.dataset_name,
            "实验名称": logger.experiment_name,
            "使用预训练权重": pretrained_str,
            "参数量(M)": logger.model_info.get("parameters", 0) / 1e6,
            "FLOPs(G)": logger.model_info.get("flops_gflops", 0),
            "模型大小(MB)": logger.model_info.get("model_size_mb", 0),
            "最佳Epoch": best_metrics.get("best_epoch", 0),
            "Precision": best_metrics.get("precision", 0),
            "Recall": best_metrics.get("recall", 0),
            "mAP@0.5": best_metrics.get("mAP@0.5", 0),
            "mAP@0.5:0.95": best_metrics.get("mAP@0.5:0.95", 0),
            "F1": best_metrics.get("F1", 0),
            "推理时间(ms)": logger.inference_metrics.get("inference_time_ms", 0),
            "端到端延迟(ms)": logger.inference_metrics.get("end_to_end_latency_ms", 0),
            "FPS": logger.inference_metrics.get("fps", 0),
            "设备": logger.inference_metrics.get("device", ""),
            "精度": logger.inference_metrics.get("precision", ""),
        }
        self.experiments.append(experiment_record)
        self._save_summary()
    def _save_summary(self):
        df = pd.DataFrame(self.experiments)
        with pd.ExcelWriter(self.summary_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name="实验汇总", index=False)
            if len(df) > 0:
                model_stats = df.groupby("模型").agg({
                    "mAP@0.5": ["mean", "max", "std"],
                    "推理时间(ms)": ["mean", "min"],
                    "FPS": ["mean", "max"]
                }).round(4)
                model_stats.to_excel(writer, sheet_name="模型统计")
                dataset_stats = df.groupby("数据集").agg({
                    "mAP@0.5": ["mean", "max", "std"],
                    "Precision": ["mean", "max"],
                    "Recall": ["mean", "max"]
                }).round(4)
                dataset_stats.to_excel(writer, sheet_name="数据集统计")
        print(f"实验汇总已更新: {self.summary_file}")
    def get_comparison_table(self) -> pd.DataFrame:
        df = pd.DataFrame(self.experiments)
        if len(df) == 0:
            return df
        columns = [
            "模型", "数据集", "参数量(M)", "FLOPs(G)",
            "mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1",
            "推理时间(ms)", "端到端延迟(ms)", "FPS"
        ]
        return df[columns].round(4)
if __name__ == "__main__":
    logger = MetricsLogger(
        experiment_name="test_exp",
        model_name="YOLOv11",
        dataset_name="blur_png"
    )
    logger.set_model_info(
        params=25000000,
        flops=8.5,
        model_size_mb=48.5
    )
    for epoch in range(10):
        logger.log_train_loss(epoch, loss=1.0 - epoch * 0.08, box_loss=0.5, cls_loss=0.3)
        logger.log_epoch_metrics(
            epoch=epoch,
            precision=0.7 + epoch * 0.02,
            recall=0.65 + epoch * 0.025,
            map50=0.6 + epoch * 0.03,
            map50_95=0.4 + epoch * 0.025
        )
    logger.log_inference_metrics(
        inference_time_ms=8.5,
        preprocess_time_ms=1.2,
        postprocess_time_ms=2.3,
        fps=100
    )
    logger.save_all()
    print("测试完成!")
