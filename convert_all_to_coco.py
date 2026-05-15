
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
from utils.yolo2coco_converter import YOLO2COCOConverter
DATASET_CONFIGS = {
    "blur_png": "configs/dataset_blur_png.yaml",
    "blur_rgb": "configs/dataset_blur_rgb.yaml",
    "dark_png": "configs/dataset_dark_png.yaml",
    "dark_rgb": "configs/dataset_dark_rgb.yaml",
}
def convert_all_datasets():
    print("=" * 60)
    print("批量转换YOLO格式到COCO格式")
    print("=" * 60)
    print()
    results = {}
    for dataset_name, yaml_path in DATASET_CONFIGS.items():
        yaml_file = project_root / yaml_path
        if not yaml_file.exists():
            print(f"⚠️  跳过 {dataset_name}: YAML文件不存在")
            continue
        print(f"\n处理数据集: {dataset_name}")
        print(f"YAML文件: {yaml_file}")
        print("-" * 60)
        try:
            converter = YOLO2COCOConverter(yaml_file)
            output_dir = converter.convert()
            results[dataset_name] = {
                "status": "success",
                "output_dir": str(output_dir)
            }
        except Exception as e:
            print(f"✗ 转换失败: {e}")
            results[dataset_name] = {
                "status": "failed",
                "error": str(e)
            }
    print("\n" + "=" * 60)
    print("转换总结")
    print("=" * 60)
    success_count = sum(1 for r in results.values() if r["status"] == "success")
    fail_count = len(results) - success_count
    print(f"成功: {success_count}/{len(results)}")
    print(f"失败: {fail_count}/{len(results)}")
    print()
    if success_count > 0:
        print("成功转换的数据集:")
        for name, result in results.items():
            if result["status"] == "success":
                print(f"  ✓ {name}: {result['output_dir']}")
    if fail_count > 0:
        print("\n失败的数据集:")
        for name, result in results.items():
            if result["status"] == "failed":
                print(f"  ✗ {name}: {result.get('error', 'Unknown error')}")
    print("\n" + "=" * 60)
    print("转换完成!")
    print("=" * 60)
    return results
if __name__ == "__main__":
    convert_all_datasets()
