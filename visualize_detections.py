
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
ultralytics_src = project_root / "ultralytics"
if ultralytics_src.exists() and (ultralytics_src / "ultralytics").exists():
    if str(ultralytics_src) not in sys.path:
        sys.path.insert(0, str(ultralytics_src))
try:
    from ultralytics import YOLO
except ImportError:
    print("错误: 请安装 ultralytics")
    print("安装命令: pip install ultralytics")
    sys.exit(1)
def find_best_model(experiment_dir: str = None, model_name: str = None, dataset_name: str = None) -> str:
    project_root = Path(__file__).parent
    if model_name and dataset_name:
        yolo_best_dir = project_root / "yolo_best"
        pattern = f"{model_name}_{dataset_name}_best.pt"
        matches = list(yolo_best_dir.glob(pattern))
        if matches:
            return str(matches[0])
    if experiment_dir:
        exp_path = Path(experiment_dir)
        best_model = exp_path / "weights" / "best.pt"
        if best_model.exists():
            return str(best_model)
    results_dir = project_root / "results" / "fish_detection"
    if results_dir.exists():
        best_models = list(results_dir.glob("**/weights/best.pt"))
        if best_models:
            if model_name:
                for model_path in best_models:
                    if model_name.lower() in str(model_path).lower():
                        return str(model_path)
            return str(max(best_models, key=lambda p: p.stat().st_mtime))
    return None
def visualize_detections(model_path: str,
                        source: str,
                        output_dir: str = None,
                        conf: float = 0.25,
                        iou: float = 0.45,
                        imgsz: int = 640,
                        save_txt: bool = False,
                        save_conf: bool = True,
                        line_width: int = 2,
                        show_labels: bool = True,
                        show_conf: bool = True,
                        device: str = "auto"):
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    print(f"正在加载模型: {model_path}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    model = YOLO(model_path)
    print(f"✓ 模型加载成功")
    if output_dir is None:
        model_dir = Path(model_path).parent
        output_dir = model_dir / "predictions"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    source_path = Path(source)
    if source_path.is_file():
        image_paths = [source_path]
    elif source_path.is_dir():
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(list(source_path.glob(f"*{ext}")))
            image_paths.extend(list(source_path.glob(f"*{ext.upper()}")))
        print(f"找到 {len(image_paths)} 张图片")
    else:
        raise ValueError(f"输入路径不存在: {source}")
    if len(image_paths) == 0:
        print("⚠️ 未找到任何图片文件")
        return
    print(f"\n开始推理 (置信度阈值: {conf}, IoU阈值: {iou})...")
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = Path(__file__).parent
    actual_output_dir = project_root / output_dir
    results = model.predict(
        source=str(source),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        save=True,
        save_txt=save_txt,
        save_conf=save_conf,
        line_width=line_width,
        show_labels=show_labels,
        show_conf=show_conf,
        project=str(actual_output_dir.parent),
        name=actual_output_dir.name,
        device=device,
        verbose=True
    )
    import shutil
    from pathlib import Path as PathLib
    possible_locations = [
        project_root / "runs" / "detect" / actual_output_dir.name,
        actual_output_dir.parent / actual_output_dir.name,
        actual_output_dir
    ]
    actual_save_dir = None
    for loc in possible_locations:
        if loc.exists() and any(loc.glob("*.jpg")) or any(loc.glob("*.png")):
            actual_save_dir = loc
            break
    if actual_save_dir and actual_save_dir != actual_output_dir:
        print(f"\n检测到结果保存在: {actual_save_dir}")
        print(f"正在复制到目标目录: {actual_output_dir}")
        actual_output_dir.mkdir(parents=True, exist_ok=True)
        image_files = list(actual_save_dir.glob("*.jpg")) + list(actual_save_dir.glob("*.png"))
        for img_file in image_files:
            shutil.copy2(img_file, actual_output_dir / img_file.name)
        if save_txt:
            txt_files = list(actual_save_dir.glob("*.txt"))
            for txt_file in txt_files:
                shutil.copy2(txt_file, actual_output_dir / txt_file.name)
        print(f"✓ 文件已复制到: {actual_output_dir}")
    elif actual_output_dir.exists():
        actual_save_dir = actual_output_dir
    print(f"\n✓ 推理完成！")
    if actual_save_dir:
        print(f"结果已保存到: {actual_save_dir}")
    else:
        print(f"结果应保存在: {actual_output_dir}")
    total_detections = 0
    for result in results:
        total_detections += len(result.boxes)
    print(f"\n统计信息:")
    print(f"  处理图片数: {len(results)}")
    print(f"  总检测数: {total_detections}")
    print(f"  平均每张图片: {total_detections/len(results):.2f} 个检测")
    return results, output_dir
def visualize_batch_comparison(model_paths: list,
                               source: str,
                               output_dir: str = None,
                               conf: float = 0.25,
                               iou: float = 0.45,
                               imgsz: int = 640,
                               device: str = "auto"):
    print(f"使用 {len(model_paths)} 个模型进行对比检测...")
    all_results = {}
    for model_path in model_paths:
        model_name = Path(model_path).stem
        print(f"\n{'='*60}")
        print(f"模型: {model_name}")
        print(f"{'='*60}")
        results, output_dir = visualize_detections(
            model_path=model_path,
            source=source,
            output_dir=output_dir / model_name if output_dir else None,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device
        )
        all_results[model_name] = results
    return all_results
def main():
    parser = argparse.ArgumentParser(
        description="YOLO检测结果可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python visualize_detections.py --model weights/best.pt --source image.jpg

  python visualize_detections.py --model weights/best.pt --source images/ --output results/

  python visualize_detections.py --model weights/best.pt --source images/ --conf 0.5

  python visualize_detections.py --model weights/model1.pt weights/model2.pt --source images/ --output comparison/
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        required=False,
        help="训练好的模型权重路径 (.pt文件)，可以指定多个模型进行对比"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="实验目录路径（会自动查找best.pt），例如: results/fish_detection/YOLO26S_dataset_blur_png_xxx"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="模型名称（用于自动查找），例如: yolo26s"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="数据集名称（用于自动查找），例如: blur-png"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="输入图片路径或目录"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录（默认保存到模型目录下的predictions文件夹）"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="置信度阈值 (默认: 0.25)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU阈值，用于NMS (默认: 0.45)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="输入图片尺寸 (默认: 640)"
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="保存检测结果为txt文件（YOLO格式）"
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="不显示标签"
    )
    parser.add_argument(
        "--no-conf",
        action="store_true",
        help="不在标签中显示置信度"
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="检测框线宽 (默认: 2)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="设备 (默认: auto)"
    )
    args = parser.parse_args()
    if args.model is None or len(args.model) == 0:
        if args.experiment:
            model_path = find_best_model(experiment_dir=args.experiment)
        elif args.model_name and args.dataset_name:
            model_path = find_best_model(model_name=args.model_name, dataset_name=args.dataset_name)
        else:
            parser.error("请指定 --model 或 --experiment 或 (--model-name 和 --dataset-name)")
        if model_path:
            print(f"自动找到模型: {model_path}")
            args.model = [model_path]
        else:
            parser.error("未找到模型，请使用 --model 指定模型路径")
    if len(args.model) == 1:
        visualize_detections(
            model_path=args.model[0],
            source=args.source,
            output_dir=args.output,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            save_txt=args.save_txt,
            save_conf=not args.no_conf,
            line_width=args.line_width,
            show_labels=not args.no_labels,
            show_conf=not args.no_conf,
            device=args.device
        )
    else:
        visualize_batch_comparison(
            model_paths=args.model,
            source=args.source,
            output_dir=args.output,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device
        )
if __name__ == "__main__":
    main()
