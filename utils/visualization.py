
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
PAPER_STYLE = {
    'figure.figsize': (8, 6),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'lines.linewidth': 2,
    'lines.markersize': 8,
}
COLORS = {
    'YOLOv11': '#2ecc71',
    'YOLOv11n': '#27ae60',
    'YOLOv11s': '#2ecc71',
    'YOLOv11m': '#1abc9c',
    'YOLOv11l': '#16a085',
    'YOLOv11x': '#0d7377',
    'RT-DETR': '#e74c3c',
    'RT-DETR-L': '#c0392b',
    'RT-DETR-X': '#e74c3c',
    'SAM3': '#9b59b6',
    'YOLO': '#3498db',
    'default': '#34495e'
}
def get_color(model_name: str) -> str:
    for key in COLORS:
        if key.lower() in model_name.lower():
            return COLORS[key]
    return COLORS['default']
def plot_pr_curves(pr_data: Dict[str, Dict],
                   save_path: Optional[str] = None,
                   title: str = "Precision-Recall Curve",
                   figsize: Tuple[int, int] = (8, 6)):
    plt.rcParams.update(PAPER_STYLE)
    fig, ax = plt.subplots(figsize=figsize)
    for model_name, data in pr_data.items():
        precision = data["precision"]
        recall = data["recall"]
        color = get_color(model_name)
        auc = np.trapz(precision, recall)
        label = f"{model_name} (AUC={auc:.3f})"
        ax.plot(recall, precision, color=color, label=label, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR曲线已保存到: {save_path}")
    return fig, ax
def plot_map_over_epochs(metrics_data: Dict[str, List[Dict]],
                         metric_key: str = "mAP@0.5",
                         save_path: Optional[str] = None,
                         title: str = "mAP@0.5",
                         figsize: Tuple[int, int] = (10, 6)):
    plt.rcParams.update(PAPER_STYLE)
    fig, ax = plt.subplots(figsize=figsize)
    for model_name, metrics in metrics_data.items():
        epochs = [m["epoch"] for m in metrics]
        values = [m[metric_key] for m in metrics]
        color = get_color(model_name)
        ax.plot(epochs, values, color=color, label=model_name, linewidth=2, marker='o', markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_key)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"曲线图已保存到: {save_path}")
    return fig, ax
def plot_loss_curves(loss_data: Dict[str, List[Dict]],
                     loss_key: str = "total_loss",
                     save_path: Optional[str] = None,
                     title: str = "Training Loss",
                     figsize: Tuple[int, int] = (10, 6)):
    plt.rcParams.update(PAPER_STYLE)
    fig, ax = plt.subplots(figsize=figsize)
    for model_name, losses in loss_data.items():
        epochs = [l["epoch"] for l in losses]
        values = [l[loss_key] for l in losses]
        color = get_color(model_name)
        ax.plot(epochs, values, color=color, label=model_name, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss曲线已保存到: {save_path}")
    return fig, ax
def plot_inference_comparison(inference_data: List[Dict],
                              save_path: Optional[str] = None,
                              title: str = "Inference Time vs mAP",
                              figsize: Tuple[int, int] = (10, 8)):
    plt.rcParams.update(PAPER_STYLE)
    fig, ax = plt.subplots(figsize=figsize)
    for data in inference_data:
        model = data["model"]
        x = data["inference_time_ms"]
        y = data["mAP"] * 100
        size = data.get("params_m", 10) * 10
        color = get_color(model)
        ax.scatter(x, y, s=size, c=color, alpha=0.7, edgecolors='white', linewidth=2)
        ax.annotate(model, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=10)
    ax.set_xlabel("Inference Time (ms)")
    ax.set_ylabel("mAP (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"推理对比图已保存到: {save_path}")
    return fig, ax
def plot_latency_comparison(latency_data: List[Dict],
                            save_path: Optional[str] = None,
                            title: str = "End-to-end Latency vs COCO AP",
                            figsize: Tuple[int, int] = (10, 8)):
    plt.rcParams.update(PAPER_STYLE)
    fig, ax = plt.subplots(figsize=figsize)
    model_groups = {}
    for data in latency_data:
        model = data["model"]
        base_model = model.split('-')[0] if '-' in model else model.rstrip('nsmxl')
        if base_model not in model_groups:
            model_groups[base_model] = []
        model_groups[base_model].append(data)
    for base_model, group in model_groups.items():
        color = get_color(base_model)
        xs = [d["latency_ms"] for d in group]
        ys = [d["ap"] for d in group]
        labels = [d.get("variant", "") for d in group]
        if len(xs) > 1:
            sorted_indices = np.argsort(xs)
            xs_sorted = [xs[i] for i in sorted_indices]
            ys_sorted = [ys[i] for i in sorted_indices]
            ax.plot(xs_sorted, ys_sorted, color=color, linewidth=2, alpha=0.7)
        ax.scatter(xs, ys, c=color, s=100, marker='o', edgecolors='white', linewidth=2, label=base_model)
        for x, y, label in zip(xs, ys, labels):
            if label:
                ax.annotate(label, (x, y), textcoords="offset points", xytext=(3, 3), fontsize=9, fontweight='bold')
    ax.set_xlabel("End-to-end Latency T4 TensorRT FP16 (ms)")
    ax.set_ylabel("COCO AP (%)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"延迟对比图已保存到: {save_path}")
    return fig, ax
def plot_model_comparison_bar(comparison_data: pd.DataFrame,
                              metrics: List[str] = ["mAP@0.5", "Precision", "Recall"],
                              save_path: Optional[str] = None,
                              title: str = "Model Comparison",
                              figsize: Tuple[int, int] = (12, 6)):
    plt.rcParams.update(PAPER_STYLE)
    fig, ax = plt.subplots(figsize=figsize)
    models = comparison_data["模型"].tolist()
    x = np.arange(len(models))
    width = 0.8 / len(metrics)
    for i, metric in enumerate(metrics):
        values = comparison_data[metric].tolist()
        offset = (i - len(metrics)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric, alpha=0.8)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图已保存到: {save_path}")
    return fig, ax
def load_metrics_from_json(json_path: str) -> Dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)
def create_paper_figures(results_dir: str, output_dir: str):
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    all_losses = {}
    all_pr_data = {}
    inference_comparison = []
    for json_file in results_path.glob("**/*_metrics.json"):
        data = load_metrics_from_json(json_file)
        model_name = data["model_name"]
        all_metrics[model_name] = data["epoch_metrics"]
        all_losses[model_name] = data["train_losses"]
        if data.get("pr_curve_data"):
            all_pr_data[model_name] = data["pr_curve_data"].get("fish", {})
        if data.get("inference_metrics"):
            best = data.get("best_metrics", {})
            inference_comparison.append({
                "model": model_name,
                "inference_time_ms": data["inference_metrics"].get("inference_time_ms", 0),
                "mAP": best.get("mAP@0.5", 0),
                "params_m": data.get("model_info", {}).get("parameters", 0) / 1e6
            })
    if all_metrics:
        plot_map_over_epochs(all_metrics, "mAP@0.5",
                            output_path / "map50_curves.png", "mAP@0.5 over Epochs")
        plot_map_over_epochs(all_metrics, "mAP@0.5:0.95",
                            output_path / "map50_95_curves.png", "mAP@0.5:0.95 over Epochs")
    if all_losses:
        plot_loss_curves(all_losses, "total_loss",
                        output_path / "loss_curves.png", "Training Loss")
    if all_pr_data:
        plot_pr_curves(all_pr_data, output_path / "pr_curves.png", "Precision-Recall Curve")
    if inference_comparison:
        plot_inference_comparison(inference_comparison,
                                 output_path / "inference_comparison.png",
                                 "Inference Time vs mAP")
    print(f"所有图表已保存到: {output_path}")
if __name__ == "__main__":
    pr_data = {
        "YOLOv11": {
            "precision": [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            "recall": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        },
        "RT-DETR": {
            "precision": [1.0, 0.92, 0.88, 0.82, 0.78, 0.72, 0.65, 0.55, 0.45, 0.35],
            "recall": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        }
    }
    plot_pr_curves(pr_data, "test_pr_curve.png")
    print("可视化测试完成!")
