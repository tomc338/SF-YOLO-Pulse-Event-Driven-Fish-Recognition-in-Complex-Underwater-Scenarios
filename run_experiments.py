
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
DATASET_CONFIGS = {
    "blur_png": "configs/dataset_blur_png.yaml",
    "blur_rgb": "configs/dataset_blur_rgb.yaml",
    "dark_png": "configs/dataset_dark_png.yaml",
    "dark_rgb": "configs/dataset_dark_rgb.yaml",
}
MODEL_CONFIGS = {
    "yolo26n": {"trainer": "yolo", "name": "yolo26n", "from_scratch": True},
    "yolo26s": {"trainer": "yolo", "name": "yolo26s", "from_scratch": True},
    "yolo26m": {"trainer": "yolo", "name": "yolo26m", "from_scratch": True},
    "yolo26l": {"trainer": "yolo", "name": "yolo26l", "from_scratch": True},
    "yolo26x": {"trainer": "yolo", "name": "yolo26x", "from_scratch": True},
    "yolo26s-haar": {"trainer": "yolo", "name": "yolo26s-haar", "from_scratch": True},
    "yolo26s-spd": {"trainer": "yolo", "name": "yolo26s-spd", "from_scratch": True},
    "yolo26s-snake": {"trainer": "yolo", "name": "yolo26s-snake", "from_scratch": True},
    "yolo26s-polarity": {"trainer": "yolo", "name": "yolo26s-polarity", "from_scratch": True},
    "yolo26s-frequency": {"trainer": "yolo", "name": "yolo26s-frequency", "from_scratch": True},
    "yolo26s-lateral": {"trainer": "yolo", "name": "yolo26s-lateral", "from_scratch": True},
    "yolo26s-graph": {"trainer": "yolo", "name": "yolo26s-graph", "from_scratch": True},
    "yolo26s-vmamba": {"trainer": "yolo", "name": "yolo26s-vmamba", "from_scratch": True},
    "yolo26s-dcnv4": {"trainer": "yolo", "name": "yolo26s-dcnv4", "from_scratch": True},
    "yolo26s-spd-enhanced": {"trainer": "yolo", "name": "yolo26s-spd-enhanced", "from_scratch": True},
    "yolo26s-seam": {"trainer": "yolo", "name": "yolo26s-seam", "from_scratch": True},
    "yolo26s-connectivity": {"trainer": "yolo", "name": "yolo26s-connectivity", "from_scratch": True},
    "yolo26s-hgblock": {"trainer": "yolo", "name": "yolo26s-hgblock", "from_scratch": True},
    "yolo26s-new": {"trainer": "yolo", "name": "yolo26s-new", "from_scratch": True},
    "yolo11n": {"trainer": "yolo", "name": "yolo11n", "from_scratch": True},
    "yolo11s": {"trainer": "yolo", "name": "yolo11s", "from_scratch": True},
    "yolo11m": {"trainer": "yolo", "name": "yolo11m", "from_scratch": True},
    "yolo11l": {"trainer": "yolo", "name": "yolo11l", "from_scratch": True},
    "yolo11x": {"trainer": "yolo", "name": "yolo11x", "from_scratch": True},
    "yolov10n": {"trainer": "yolo", "name": "yolov10n", "from_scratch": True},
    "yolov10s": {"trainer": "yolo", "name": "yolov10s", "from_scratch": True},
    "yolov10m": {"trainer": "yolo", "name": "yolov10m", "from_scratch": True},
    "yolov8n": {"trainer": "yolo", "name": "yolov8n", "from_scratch": True},
    "yolov8s": {"trainer": "yolo", "name": "yolov8s", "from_scratch": True},
    "yolov8m": {"trainer": "yolo", "name": "yolov8m", "from_scratch": True},

    "rtdetr-s": {"trainer": "rtdetr", "name": "rtdetr-s", "from_scratch": False},
    "rtdetr-m": {"trainer": "rtdetr", "name": "rtdetr-m", "from_scratch": False},
    "rtdetr-l": {"trainer": "rtdetr", "name": "rtdetr-l", "from_scratch": False},
    "rtdetr-x": {"trainer": "rtdetr", "name": "rtdetr-x", "from_scratch": False},

    "detr-r18": {"trainer": "detr", "name": "detr", "backbone": "resnet18", "from_scratch": True},
    "detr-r50": {"trainer": "detr", "name": "detr", "backbone": "resnet50", "from_scratch": True},

    "dino-r50": {"trainer": "dino", "name": "dino", "backbone": "resnet50", "scale": "4scale", "from_scratch": True},

    "fasterrcnn-mobilenet": {"trainer": "fasterrcnn", "name": "faster_rcnn", "backbone": "mobilenet_v3_large_fpn", "from_scratch": True},
    "fasterrcnn-r50": {"trainer": "fasterrcnn", "name": "faster_rcnn", "backbone": "resnet50_fpn", "from_scratch": True},

    "un-detr": {"trainer": "undetr", "name": "un-detr", "from_scratch": True},

    "ia-yolo-n": {"trainer": "iayolo", "name": "ia-yolo", "size": "n", "from_scratch": True},
    "ia-yolo-s": {"trainer": "iayolo", "name": "ia-yolo", "size": "s", "from_scratch": True},
    "ia-yolo-m": {"trainer": "iayolo", "name": "ia-yolo", "size": "m", "from_scratch": True},

    "yolo11n-biformer": {"trainer": "yolo_biformer", "name": "yolo11n", "from_scratch": True},
    "yolo11s-biformer": {"trainer": "yolo_biformer", "name": "yolo11s", "from_scratch": True},
    "yolo26n-biformer": {"trainer": "yolo_biformer", "name": "yolo26n", "from_scratch": True},
    "yolo26s-biformer": {"trainer": "yolo_biformer", "name": "yolo26s", "from_scratch": True},
}
def get_trainer(model_config: Dict, data_yaml: str, device: str = "auto", pretrained: bool = True):
    trainer_type = model_config["trainer"]
    model_name = model_config["name"]
    supports_from_scratch = model_config.get("from_scratch", True)
    if not pretrained and not supports_from_scratch:
        raise ValueError(
            f"模型 {model_name} 不支持完全从头训练。\n"
            f"科研公平对比建议：\n"
            f"1. 使用支持从头训练的模型: yolo11n, detr-r18, fasterrcnn-mobilenet\n"
            f"2. 或者所有模型都使用预训练权重 (--pretrained)"
        )
    if trainer_type == "yolo":
        from trainers.yolo_trainer import YOLOTrainer
        return YOLOTrainer(
            model_name=model_name,
            data_yaml=data_yaml,
            device=device,
            pretrained=pretrained
        )
    elif trainer_type == "rtdetr":
        from trainers.rtdetr_trainer import RTDETRTrainer
        return RTDETRTrainer(
            model_name=model_name,
            data_yaml=data_yaml,
            device=device,
            pretrained=pretrained
        )
    elif trainer_type == "detr":
        from trainers.detr_trainer import DETRTrainer
        backbone = model_config.get("backbone", "resnet50")
        return DETRTrainer(
            model_name=model_name,
            backbone=backbone,
            data_yaml=data_yaml,
            device=device,
            pretrained=pretrained
        )
    elif trainer_type == "dino":
        from trainers.dino_trainer import DINOTrainer
        backbone = model_config.get("backbone", "resnet50")
        scale = model_config.get("scale", "4scale")
        return DINOTrainer(
            model_name=model_name,
            backbone=backbone,
            scale=scale,
            data_yaml=data_yaml,
            device=device,
            pretrained=pretrained
        )
    elif trainer_type == "fasterrcnn":
        from trainers.fasterrcnn_trainer import FasterRCNNTrainer
        backbone = model_config.get("backbone", "resnet50_fpn")
        return FasterRCNNTrainer(
            model_name=model_name,
            backbone=backbone,
            data_yaml=data_yaml,
            device=device,
            pretrained=pretrained
        )
    elif trainer_type == "undetr":
        from trainers.undetr_trainer import UNDETRTrainer
        return UNDETRTrainer(
            model_name=model_name,
            data_yaml=data_yaml,
            device=device,
            pretrained=pretrained
        )
    elif trainer_type == "iayolo":
        from trainers.iayolo_trainer import IAYOLOTrainer
        model_size = model_config.get("size", "n")
        return IAYOLOTrainer(
            model_name=model_name,
            model_size=model_size,
            data_yaml=data_yaml,
            device=device,
            pretrained=pretrained
        )
    elif trainer_type == "yolo_biformer":
        from trainers.yolo_biformer_trainer import YOLOBiFormerTrainer
        return YOLOBiFormerTrainer(
            model_name=model_name,
            data_yaml=data_yaml,
            device=device,
            pretrained=pretrained,
            use_biformer_neck=True
        )
    else:
        raise ValueError(f"未知的训练器类型: {trainer_type}")
def run_single_experiment(
    model_name: str,
    dataset_name: str,
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "auto",
    benchmark: bool = True,
    pretrained: bool = True,
    **kwargs
) -> Dict:
    print(f"\n{'='*70}")
    print(f"实验: {model_name} on {dataset_name}")
    print(f"{'='*70}")
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"未知的数据集: {dataset_name}. 可用: {list(DATASET_CONFIGS.keys())}")
    data_yaml_path = project_root / DATASET_CONFIGS[dataset_name]
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"数据集配置文件不存在: {data_yaml_path}")
    data_yaml = str(data_yaml_path.resolve())
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"未知的模型: {model_name}. 可用: {list(MODEL_CONFIGS.keys())}")
    model_config = MODEL_CONFIGS[model_name]
    new_models = ['spd', 'snake', 'polarity', 'frequency', 'lateral', 'graph', 'vmamba', 'dcnv4', 'spd-enhanced']
    is_new_model = any(nm in model_name for nm in new_models)
    user_explicitly_set_epochs = kwargs.pop('epochs_explicit', False)
    if is_new_model and epochs == 100 and not user_explicitly_set_epochs:
        epochs = 50
        print(f"⚠️  检测到新模型，使用默认epochs=50（可通过--epochs参数修改）")
    trainer = get_trainer(model_config, data_yaml, device, pretrained)
    trainer.load_model()
    results = trainer.train(
        epochs=epochs,
        batch_size=batch_size,
        imgsz=imgsz,
        **kwargs
    )
    if benchmark:
        trainer.benchmark_inference()
    trainer.save_results()
    from utils.metrics_logger import ExperimentTracker
    tracker = ExperimentTracker("fish_detection", str(project_root / "results"))
    tracker.add_experiment(trainer.logger)
    return results
def run_all_experiments(
    datasets: List[str] = None,
    models: List[str] = None,
    epochs: int = 100,
    batch_size: int = 16,
    device: str = "auto",
    workers: int = 0,
    **kwargs
):
    if datasets is None:
        datasets = list(DATASET_CONFIGS.keys())
    if models is None:
        models = list(MODEL_CONFIGS.keys())
    total = len(datasets) * len(models)
    current = 0
    results_summary = []
    for dataset in datasets:
        for model in models:
            current += 1
            print(f"\n{'#'*70}")
            print(f"# 进度: {current}/{total}")
            print(f"# 模型: {model}")
            print(f"# 数据集: {dataset}")
            print(f"{'#'*70}")
            try:
                result = run_single_experiment(
                    model_name=model,
                    dataset_name=dataset,
                    epochs=epochs,
                    batch_size=batch_size,
                    device=device,
                    **kwargs
                )
                results_summary.append({
                    "model": model,
                    "dataset": dataset,
                    "status": "success",
                    "result": result
                })
            except Exception as e:
                print(f"实验失败: {e}")
                results_summary.append({
                    "model": model,
                    "dataset": dataset,
                    "status": "failed",
                    "error": str(e)
                })
    print(f"\n{'='*70}")
    print("实验总结")
    print(f"{'='*70}")
    success = sum(1 for r in results_summary if r["status"] == "success")
    failed = sum(1 for r in results_summary if r["status"] == "failed")
    print(f"成功: {success}/{total}")
    print(f"失败: {failed}/{total}")
    if failed > 0:
        print("\n失败的实验:")
        for r in results_summary:
            if r["status"] == "failed":
                print(f"  - {r['model']} on {r['dataset']}: {r['error']}")
    return results_summary
def benchmark_only(
    model_name: str,
    device: str = "auto",
    num_runs: int = 100
) -> Dict:
    print(f"\n测试 {model_name} 推理性能...")
    model_config = MODEL_CONFIGS[model_name]
    data_yaml = str(project_root / DATASET_CONFIGS["blur_png"])
    trainer = get_trainer(model_config, data_yaml, device)
    trainer.load_model()
    results = trainer.benchmark_inference(num_runs=num_runs)
    return results
def generate_visualizations(results_dir: str = None, output_dir: str = None):
    from utils.visualization import create_paper_figures
    if results_dir is None:
        results_dir = str(project_root / "results")
    if output_dir is None:
        output_dir = str(project_root / "figures")
    print(f"\n生成可视化图表...")
    print(f"结果目录: {results_dir}")
    print(f"输出目录: {output_dir}")
    create_paper_figures(results_dir, output_dir)
    print("可视化图表生成完成!")
def main():
    parser = argparse.ArgumentParser(
        description="鱼类检测实验脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_experiments.py --model yolo11n --dataset blur_png --epochs 100

  python run_experiments.py --model yolo11n yolo11s rtdetr-l --dataset blur_png

  python run_experiments.py --model yolo11n --all-datasets

  python run_experiments.py --all --epochs 50

  python run_experiments.py --model yolo11n --benchmark-only

  python run_experiments.py --visualize

可用模型:
  YOLO: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
        yolov10n, yolov10s, yolov10m
        yolov8n, yolov8s, yolov8m
  RT-DETR: rtdetr-l, rtdetr-x

可用数据集:
  blur_png, blur_rgb, dark_png, dark_rgb
        """
    )
    parser.add_argument("--model", "-m", nargs="+", type=str,
                        help="模型名称 (可多选)")
    parser.add_argument("--dataset", "-d", nargs="+", type=str,
                        help="数据集名称 (可多选)")
    parser.add_argument("--all", action="store_true",
                        help="运行所有模型和数据集的组合")
    parser.add_argument("--all-models", action="store_true",
                        help="使用所有模型")
    parser.add_argument("--all-datasets", action="store_true",
                        help="使用所有数据集")
    parser.add_argument("--epochs", "-e", type=int, default=100,
                        help="训练轮数 (默认: 100)")
    parser.add_argument("--batch-size", "-b", type=int, default=8,
                        help="批次大小 (默认: 8, 显存不足时可减小到4)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="输入图像尺寸 (默认: 640, 可减小到416或512)")
    parser.add_argument("--device", type=str, default="auto",
                        help="设备 (auto, cuda, cpu)")
    parser.add_argument("--workers", type=int, default=0,
                        help="数据加载线程数 (默认0避免内存问题)")
    parser.add_argument("--no-ema", action="store_true",
                        help="禁用EMA以节省显存")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="使用预训练权重 (默认: True)")
    parser.add_argument("--from-scratch", dest="pretrained", action="store_false",
                        help="从头训练，不使用预训练权重 (科研公平对比推荐)")
    parser.add_argument("--benchmark-only", action="store_true",
                        help="仅测试推理性能")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="跳过推理性能测试")
    parser.add_argument("--visualize", action="store_true",
                        help="生成可视化图表")
    parser.add_argument("--list-models", action="store_true",
                        help="列出所有可用模型")
    parser.add_argument("--list-datasets", action="store_true",
                        help="列出所有可用数据集")
    args = parser.parse_args()
    if args.list_models:
        print("可用模型:")
        for name, config in MODEL_CONFIGS.items():
            print(f"  {name} ({config['trainer']})")
        return
    if args.list_datasets:
        print("可用数据集:")
        for name, path in DATASET_CONFIGS.items():
            print(f"  {name}: {path}")
        return
    if args.visualize:
        generate_visualizations()
        return
    if args.all or args.all_models:
        models = list(MODEL_CONFIGS.keys())
    elif args.model:
        models = args.model
    else:
        models = None
    if args.all or args.all_datasets:
        datasets = list(DATASET_CONFIGS.keys())
    elif args.dataset:
        datasets = args.dataset
    else:
        datasets = None
    if args.benchmark_only:
        if not models:
            print("错误: 请指定模型 (--model)")
            return
        for model in models:
            benchmark_only(model, args.device)
        return
    if not models or not datasets:
        if args.all:
            run_all_experiments(
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                workers=args.workers
            )
        else:
            print("错误: 请指定模型和数据集，或使用 --all 运行所有实验")
            print("使用 --help 查看帮助")
        return
    for dataset in datasets:
        for model in models:
            run_single_experiment(
                model_name=model,
                dataset_name=dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                imgsz=args.imgsz,
                device=args.device,
                benchmark=not args.no_benchmark,
                pretrained=args.pretrained,
                workers=args.workers,
                epochs_explicit=True,
                ema=not args.no_ema
            )
    print("\n所有实验完成!")
    print(f"结果保存在: {project_root / 'results'}")
if __name__ == "__main__":
    main()
