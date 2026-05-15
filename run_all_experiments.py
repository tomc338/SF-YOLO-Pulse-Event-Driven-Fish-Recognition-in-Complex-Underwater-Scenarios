
import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
from run_experiments import (
    run_single_experiment,
    DATASET_CONFIGS,
    MODEL_CONFIGS,
    generate_visualizations
)
from utils.metrics_logger import ExperimentTracker
DEFAULT_MODELS = [
    "yolo11n",

    "detr-r18",

    "fasterrcnn-mobilenet",


]
ALL_DATASETS = list(DATASET_CONFIGS.keys())
def run_all_experiments(
    models: list = None,
    datasets: list = None,
    epochs: int = 100,
    batch_size: int = 2,
    imgsz: int = 640,
    device: str = "auto",
    pretrained: bool = False,
    workers: int = 0,
    skip_benchmark: bool = False,
):
    if models is None:
        models = DEFAULT_MODELS.copy()
    else:
        models = models.copy()
    if datasets is None:
        datasets = ALL_DATASETS
    current = 0
    start_time = time.time()
    results_summary = []
    failed_experiments = []
    valid_models = []
    skipped_models = []
    print("=" * 80)
    print("鱼类检测全量实验 - 模型检查")
    print("=" * 80)
    for m in models:
        if m not in MODEL_CONFIGS:
            print(f"⚠️  未知模型: {m}，跳过")
            skipped_models.append((m, "未知模型"))
            continue
        config = MODEL_CONFIGS[m]
        supports_from_scratch = config.get("from_scratch", True)
        if not pretrained and not supports_from_scratch:
            print(f"⚠️  {m} 不支持从头训练，将跳过")
            skipped_models.append((m, "不支持从头训练"))
            continue
        if config["trainer"] in ["undetr", "dino"]:
            try:
                sys.path.insert(0, str(project_root / "DINO-main" / "models" / "dino" / "ops"))
                from functions import MSDeformAttnFunction
                valid_models.append(m)
            except ImportError:
                print(f"⚠️  {m} 需要编译CUDA扩展MultiScaleDeformableAttention，将跳过")
                print(f"   编译方法: cd DINO-main/models/dino/ops && python setup.py build install")
                skipped_models.append((m, "需要编译CUDA扩展"))
                continue
        else:
            valid_models.append(m)
    models = valid_models
    total_experiments = len(models) * len(datasets)
    print()
    print("=" * 80)
    print("实验配置")
    print("=" * 80)
    print(f"有效模型数量: {len(models)}")
    print(f"数据集数量: {len(datasets)}")
    print(f"总实验数: {total_experiments}")
    print(f"每个实验轮数: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"使用预训练权重: {'是' if pretrained else '否 (从头训练)'}")
    print("=" * 80)
    print()
    print("待运行的模型:")
    for m in models:
        config = MODEL_CONFIGS.get(m, {})
        from_scratch_tag = "[从头训练✓]" if config.get("from_scratch", True) else "[仅预训练]"
        print(f"  - {m} {from_scratch_tag}")
    if skipped_models:
        print("\n将跳过的模型:")
        for m, reason in skipped_models:
            print(f"  - {m}: {reason}")
    print()
    print("待运行的数据集:")
    for d in datasets:
        print(f"  - {d}")
    print()
    if len(models) == 0:
        print("⚠️  没有可运行的模型！")
        print("  如果使用 --no-pretrained (从头训练)，请选择支持从头训练的模型:")
        print("  - YOLO系列: yolo11n, yolo11s, yolov8n, etc.")
        print("  - DETR: detr-r18, detr-r50")
        print("  - DINO: dino-r50")
        print("  - Faster R-CNN: fasterrcnn-mobilenet, fasterrcnn-r50")
        return
    for dataset in datasets:
        print(f"\n{'#' * 80}")
        print(f"# 数据集: {dataset}")
        print(f"{'#' * 80}")
        for model in models:
            current += 1
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
            except:
                pass
            print(f"\n{'=' * 60}")
            print(f"实验进度: {current}/{total_experiments}")
            print(f"模型: {model}")
            print(f"数据集: {dataset}")
            print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'=' * 60}")
            try:
                result = run_single_experiment(
                    model_name=model,
                    dataset_name=dataset,
                    epochs=epochs,
                    batch_size=batch_size,
                    imgsz=imgsz,
                    device=device,
                    benchmark=not skip_benchmark,
                    pretrained=pretrained,
                    workers=workers,
                )
                results_summary.append({
                    "model": model,
                    "dataset": dataset,
                    "status": "success",
                    "best_map50": result.get("best_metrics", {}).get("mAP@0.5", 0)
                })
                print(f"✓ 实验完成: {model} on {dataset}")
            except ValueError as e:
                error_msg = str(e)
                if "不支持完全从头训练" in error_msg or "RT-DETR" in error_msg:
                    print(f"⚠️  跳过实验: {model} on {dataset}")
                    print(f"  原因: RT-DETR不支持完全从头训练，为保证科研公平性已跳过")
                    print(f"  建议: 使用 --pretrained 参数，或从模型列表中移除RT-DETR")
                    results_summary.append({
                        "model": model,
                        "dataset": dataset,
                        "status": "skipped",
                        "reason": "RT-DETR不支持完全从头训练（科研公平性）"
                    })
                else:
                    raise
            except Exception as e:
                print(f"✗ 实验失败: {model} on {dataset}")
                print(f"  错误: {str(e)}")
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
                failed_experiments.append({
                    "model": model,
                    "dataset": dataset,
                    "error": str(e)
                })
                results_summary.append({
                    "model": model,
                    "dataset": dataset,
                    "status": "failed",
                    "error": str(e)
                })
            elapsed = time.time() - start_time
            avg_time = elapsed / current
            remaining = avg_time * (total_experiments - current)
            print(f"\n已用时间: {elapsed/3600:.2f}小时")
            print(f"预计剩余: {remaining/3600:.2f}小时")
    total_time = time.time() - start_time
    print("\n")
    print("=" * 80)
    print("实验总结")
    print("=" * 80)
    success_count = sum(1 for r in results_summary if r["status"] == "success")
    fail_count = len(failed_experiments)
    skipped_count = sum(1 for r in results_summary if r.get("status") == "skipped")
    print(f"总实验数: {total_experiments}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    if skipped_count > 0:
        print(f"跳过: {skipped_count} (RT-DETR不支持完全从头训练)")
    print(f"总用时: {total_time/3600:.2f}小时")
    if failed_experiments:
        print("\n失败的实验:")
        for exp in failed_experiments:
            print(f"  - {exp['model']} on {exp['dataset']}: {exp['error'][:50]}...")
    print("\n成功实验的mAP@0.5结果:")
    print("-" * 60)
    print(f"{'模型':<20} {'数据集':<15} {'mAP@0.5':<10}")
    print("-" * 60)
    for r in results_summary:
        if r["status"] == "success":
            print(f"{r['model']:<20} {r['dataset']:<15} {r.get('best_map50', 0):.4f}")
    print("-" * 60)
    print("\n正在生成可视化图表...")
    try:
        generate_visualizations()
        print("✓ 图表生成完成")
    except Exception as e:
        print(f"✗ 图表生成失败: {e}")
    print("\n所有实验完成!")
    print(f"结果保存在: {project_root / 'results'}")
    return results_summary
def main():
    parser = argparse.ArgumentParser(
        description="一键运行所有鱼类检测实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_all_experiments.py --epochs 100

  python run_all_experiments.py --models yolo11n rtdetr-l --epochs 50

  python run_all_experiments.py --datasets blur_png dark_png --epochs 100

  python run_all_experiments.py --pretrained --epochs 50

  python run_all_experiments.py --epochs 10 --batch-size 4

默认运行的模型:
  - yolo11n (YOLO11 Nano)
  - yolo11s (YOLO11 Small)
  - rtdetr-l (RT-DETR Large)
  - un-detr (UN-DETR 水下专用)
  - ia-yolo-n (IA-YOLO 自适应增强)

数据集:
  - blur_png (模糊场景PNG)
  - blur_rgb (模糊场景RGB)
  - dark_png (暗光场景PNG)
  - dark_rgb (暗光场景RGB)
        """
    )
    parser.add_argument("--models", "-m", nargs="+", type=str, default=None,
                        help="要运行的模型列表 (默认: 核心对比模型)")
    parser.add_argument("--datasets", "-d", nargs="+", type=str, default=None,
                        help="要运行的数据集列表 (默认: 全部)")
    parser.add_argument("--epochs", "-e", type=int, default=100,
                        help="训练轮数 (默认: 100)")
    parser.add_argument("--batch-size", "-b", type=int, default=2,
                        help="批次大小 (默认: 2, 适配8GB显存)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="图像尺寸 (默认: 640)")
    parser.add_argument("--device", type=str, default="auto",
                        help="设备 (auto, cuda, cpu)")
    parser.add_argument("--pretrained", action="store_true",
                        help="使用预训练权重")
    parser.add_argument("--from-scratch", dest="pretrained", action="store_false",
                        help="从头训练 (默认)")
    parser.add_argument("--workers", type=int, default=0,
                        help="数据加载线程数 (默认: 0)")
    parser.add_argument("--skip-benchmark", action="store_true",
                        help="跳过推理性能测试")
    parser.add_argument("--list-models", action="store_true",
                        help="列出所有可用模型")
    parser.add_argument("--list-datasets", action="store_true",
                        help="列出所有可用数据集")
    parser.set_defaults(pretrained=False)
    args = parser.parse_args()
    if args.list_models:
        print("可用模型:")
        for name, config in MODEL_CONFIGS.items():
            trainer = config.get("trainer", "unknown")
            print(f"  {name:<15} ({trainer})")
        return
    if args.list_datasets:
        print("可用数据集:")
        for name, path in DATASET_CONFIGS.items():
            print(f"  {name:<15} -> {path}")
        return
    if args.models:
        for m in args.models:
            if m not in MODEL_CONFIGS:
                print(f"错误: 未知模型 '{m}'")
                print(f"可用模型: {list(MODEL_CONFIGS.keys())}")
                return
    if args.datasets:
        for d in args.datasets:
            if d not in DATASET_CONFIGS:
                print(f"错误: 未知数据集 '{d}'")
                print(f"可用数据集: {list(DATASET_CONFIGS.keys())}")
                return
    run_all_experiments(
        models=args.models,
        datasets=args.datasets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        pretrained=args.pretrained,
        workers=args.workers,
        skip_benchmark=args.skip_benchmark,
    )
if __name__ == "__main__":
    main()
