
import json
import os
import shutil
import sys
import time
from pathlib import Path
import cv2
from tqdm import tqdm
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
class SegYOLO2COCOSingleConverter:
    def __init__(self, dataset_dir, image_type='png', output_dir=None):
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {dataset_dir}")
        self.image_type = image_type.lower()
        if self.image_type == 'rgb':
            self.image_dir = self.dataset_dir / 'rgb'
            self.label_dir = self.dataset_dir / 'rgb_label'
        else:
            self.image_dir = self.dataset_dir / 'png'
            self.label_dir = self.dataset_dir / 'png_label'
        if not self.image_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"标签目录不存在: {self.label_dir}")
        self._build_label_mapping()
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.dataset_dir / f"COCO_format_{self.image_type}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.coco_images_dir = "images"
        self.coco_annotation = "annotations"
        (self.output_dir / self.coco_images_dir).mkdir(exist_ok=True)
        (self.output_dir / self.coco_annotation).mkdir(exist_ok=True)
        self.coco_json = self.output_dir / self.coco_annotation / 'instances_all.json'
        self.type = 'instances'
        self.categories = [{
            'supercategory': 'fish',
            'id': 1,
            'name': 'fish',
        }]
        self.annotation_id = 1
        cur_year = time.strftime('%Y', time.localtime(time.time()))
        self.info = {
            'year': int(cur_year),
            'version': '1.0',
            'description': f'Fish Detection Dataset - {self.dataset_dir.name} ({self.image_type})',
            'date_created': cur_year,
        }
        self.licenses = [{
            'id': 1,
            'name': 'Apache License v2.0',
            'url': 'https://github.com/RapidAI/YOLO2COCO/LICENSE',
        }]
    def _build_label_mapping(self):
        self.label_mapping = {}
        prefix_suffix_png = f"_{self.image_type}.rf."
        prefix_suffix_jpg = "_jpg.rf."
        for label_file in self.label_dir.glob("*.txt"):
            label_stem = label_file.stem.strip()
            if prefix_suffix_png in label_stem:
                image_stem = label_stem.split(prefix_suffix_png)[0]
                self.label_mapping[image_stem] = label_file
            elif prefix_suffix_jpg in label_stem:
                image_stem = label_stem.split(prefix_suffix_jpg)[0]
                self.label_mapping[image_stem] = label_file
            else:
                self.label_mapping[label_stem] = label_file
                if label_stem != label_stem.rstrip():
                    self.label_mapping[label_stem.rstrip()] = label_file
    def get_image_files(self):
        IMG_FORMATS = ['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp']
        image_files = []
        for ext in IMG_FORMATS:
            image_files.extend(list(self.image_dir.glob(f'*.{ext}')))
            image_files.extend(list(self.image_dir.glob(f'*.{ext.upper()}')))
        return sorted(image_files)
    def get_label_path(self, image_path):
        image_stem = image_path.stem.strip()
        if image_stem in self.label_mapping:
            return self.label_mapping[image_stem]
        if image_stem != image_stem.rstrip():
            image_stem_clean = image_stem.rstrip()
            return self.label_mapping.get(image_stem_clean, None)
        return None
    def read_annotation(self, txt_file, img_id, height, width):
        annotation = []
        if not txt_file.exists():
            return annotation
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for label_info in lines:
            label_info = label_info.strip().split()
            if len(label_info) < 5:
                continue
            category_id, vertex_info = label_info[0], label_info[1:]
            try:
                cx, cy, w, h = [float(i) for i in vertex_info]
            except ValueError:
                print(f"警告: 标注格式错误 {txt_file}: {label_info}")
                continue
            cx = cx * width
            cy = cy * height
            box_w = w * width
            box_h = h * height
            x0 = max(cx - box_w / 2, 0)
            y0 = max(cy - box_h / 2, 0)
            x1 = min(x0 + box_w, width)
            y1 = min(y0 + box_h, height)
            box_w = x1 - x0
            box_h = y1 - y0
            if box_w <= 0 or box_h <= 0:
                print(f"警告: 跳过无效bbox (w={box_w:.2f}, h={box_h:.2f}) in {txt_file}")
                continue
            segmentation = [[x0, y0, x1, y0, x1, y1, x0, y1]]
            bbox = [x0, y0, box_w, box_h]
            area = box_w * box_h
            annotation.append({
                'segmentation': segmentation,
                'area': area,
                'iscrowd': 0,
                'image_id': img_id,
                'bbox': bbox,
                'category_id': 1,
                'id': self.annotation_id,
            })
            self.annotation_id += 1
        return annotation
    def gen_dataset(self, img_paths, target_img_path, target_json):
        images = []
        annotations = []
        img_id = 0
        skipped_count = 0
        for img_path in tqdm(img_paths, desc="转换数据集"):
            img_path = Path(img_path)
            if not img_path.exists():
                skipped_count += 1
                continue
            label_path = self.get_label_path(img_path)
            if label_path is None or not label_path.exists():
                skipped_count += 1
                continue
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            if not lines:
                skipped_count += 1
                continue
            img_id += 1
            imgsrc = cv2.imread(str(img_path))
            if imgsrc is None:
                img_id -= 1
                skipped_count += 1
                continue
            height, width = imgsrc.shape[:2]
            dest_file_name = img_path.name
            if img_path.suffix.lower() not in ['.jpg', '.jpeg']:
                dest_file_name = img_path.stem + '.jpg'
            save_img_path = target_img_path / dest_file_name
            if img_path.suffix.lower() in ['.jpg', '.jpeg']:
                shutil.copyfile(img_path, save_img_path)
            else:
                cv2.imwrite(str(save_img_path), imgsrc)
            images.append({
                'date_captured': self.info['date_created'],
                'file_name': dest_file_name,
                'id': img_id,
                'height': height,
                'width': width,
            })
            new_anno = self.read_annotation(label_path, img_id, height, width)
            annotations.extend(new_anno)
        json_data = {
            'info': self.info,
            'images': images,
            'licenses': self.licenses,
            'type': self.type,
            'annotations': annotations,
            'categories': self.categories,
        }
        with open(target_json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"[OK] 数据集转换完成: {len(images)}张图像, {len(annotations)}个标注")
        if skipped_count > 0:
            print(f"  (跳过{skipped_count}个无效/无标注图像)")
        return len(images), len(annotations)
    def convert(self):
        print(f"开始转换YOLO格式到COCO格式...")
        print(f"数据集目录: {self.dataset_dir}")
        print(f"图像类型: {self.image_type}")
        print(f"输出路径: {self.output_dir}")
        print(f"类别数: 1, 类别: ['fish']")
        print()
        all_image_files = self.get_image_files()
        print(f"找到 {len(all_image_files)} 张图像")
        if len(all_image_files) == 0:
            print("错误: 未找到图像文件")
            return None
        images_dest_dir = self.output_dir / self.coco_images_dir
        image_count, anno_count = self.gen_dataset(
            all_image_files, images_dest_dir, self.coco_json
        )
        print()
        print("=" * 60)
        print("转换完成!")
        print(f"输出目录: {self.output_dir}")
        print(f"图像数量: {image_count}张")
        print(f"标注数量: {anno_count}个")
        print(f"JSON文件: {self.coco_json}")
        print("=" * 60)
        return self.output_dir
DATASET_CONFIGS = {
    "blur_seg_png": ("dataset/blur_seg", "png"),
    "blur_seg_rgb": ("dataset/blur_seg", "rgb"),
    "dark_seg_png": ("dataset/dark_seg", "png"),
    "dark_seg_rgb": ("dataset/dark_seg", "rgb"),
}
def convert_all_datasets():
    print("=" * 60)
    print("批量转换YOLO格式到COCO格式（不划分训练集和验证集）")
    print("=" * 60)
    print()
    results = {}
    for dataset_name, (dataset_path, image_type) in DATASET_CONFIGS.items():
        full_path = project_root / dataset_path
        if not full_path.exists():
            print(f"⚠️  跳过 {dataset_name}: 目录不存在")
            continue
        print(f"\n处理数据集: {dataset_name}")
        print(f"路径: {full_path}")
        print(f"图像类型: {image_type}")
        print("-" * 60)
        try:
            converter = SegYOLO2COCOSingleConverter(full_path, image_type)
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
