
import json
import os
import shutil
import time
from pathlib import Path
import cv2
from tqdm import tqdm
class SegYOLO2COCOConverter:
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
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.dataset_dir / f"COCO_format_{self.image_type}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.coco_train = "train2017"
        self.coco_val = "val2017"
        self.coco_annotation = "annotations"
        (self.output_dir / self.coco_train).mkdir(exist_ok=True)
        (self.output_dir / self.coco_val).mkdir(exist_ok=True)
        (self.output_dir / self.coco_annotation).mkdir(exist_ok=True)
        self.coco_train_json = self.output_dir / self.coco_annotation /\
            f'instances_{self.coco_train}.json'
        self.coco_val_json = self.output_dir / self.coco_annotation /\
            f'instances_{self.coco_val}.json'
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
            'description': f'Fish Detection Dataset - {self.dataset_dir.name}',
            'date_created': cur_year,
        }
        self.licenses = [{
            'id': 1,
            'name': 'Apache License v2.0',
            'url': 'https://github.com/RapidAI/YOLO2COCO/LICENSE',
        }]
    def get_image_files(self):
        IMG_FORMATS = ['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp']
        image_files = []
        for ext in IMG_FORMATS:
            image_files.extend(list(self.image_dir.glob(f'*.{ext}')))
            image_files.extend(list(self.image_dir.glob(f'*.{ext.upper()}')))
        return sorted(image_files)
    def get_label_path(self, image_path):
        label_name = image_path.stem + '.txt'
        label_path = self.label_dir / label_name
        return label_path
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
    def gen_dataset(self, img_paths, target_img_path, target_json, mode, split_ratio=0.8):
        images = []
        annotations = []
        img_id = 0
        skipped_count = 0
        for img_path in tqdm(img_paths, desc=f"转换{mode}集"):
            label_path = self.get_label_path(img_path)
            if not img_path.exists():
                skipped_count += 1
                continue
            if not label_path.exists():
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
        print(f"[OK] {mode}集转换完成: {len(images)}张图像, {len(annotations)}个标注")
        if skipped_count > 0:
            print(f"  (跳过{skipped_count}个无效/无标注图像)")
        return len(images), len(annotations)
    def convert(self, train_ratio=0.8):
        print(f"开始转换YOLO格式到COCO格式...")
        print(f"数据集目录: {self.dataset_dir}")
        print(f"图像类型: {self.image_type}")
        print(f"输出路径: {self.output_dir}")
        print()
        all_image_files = self.get_image_files()
        print(f"找到 {len(all_image_files)} 张图像")
        if len(all_image_files) == 0:
            print("错误: 未找到图像文件")
            return None
        split_idx = int(len(all_image_files) * train_ratio)
        train_files = all_image_files[:split_idx]
        val_files = all_image_files[split_idx:]
        print(f"训练集: {len(train_files)}张图像 ({train_ratio*100:.1f}%)")
        print(f"验证集: {len(val_files)}张图像 ({(1-train_ratio)*100:.1f}%)")
        print()
        train_dest_dir = self.output_dir / self.coco_train
        train_count, train_anno_count = self.gen_dataset(
            train_files, train_dest_dir, self.coco_train_json, 'train'
        )
        val_dest_dir = self.output_dir / self.coco_val
        val_count, val_anno_count = self.gen_dataset(
            val_files, val_dest_dir, self.coco_val_json, 'val'
        )
        print()
        print("=" * 60)
        print("转换完成!")
        print(f"输出目录: {self.output_dir}")
        print(f"训练集: {train_count}张图像, {train_anno_count}个标注")
        print(f"验证集: {val_count}张图像, {val_anno_count}个标注")
        print("=" * 60)
        return self.output_dir
def convert_seg_datasets():
    base_dir = Path(__file__).parent
    datasets = [
        ('dataset/blur_seg', 'blur_seg'),
        ('dataset/dark_seg', 'dark_seg'),
    ]
    image_types = ['png', 'rgb']
    print("=" * 60)
    print("批量转换YOLO格式到COCO格式")
    print("=" * 60)
    print()
    results = {}
    for dataset_path, dataset_name in datasets:
        full_path = base_dir / dataset_path
        if not full_path.exists():
            print(f"⚠️  跳过 {dataset_name}: 目录不存在")
            continue
        for img_type in image_types:
            key = f"{dataset_name}_{img_type}"
            print(f"\n处理数据集: {key}")
            print(f"路径: {full_path}")
            print("-" * 60)
            try:
                converter = SegYOLO2COCOConverter(full_path, img_type)
                output_dir = converter.convert()
                results[key] = {
                    "status": "success",
                    "output_dir": str(output_dir)
                }
            except Exception as e:
                print(f"✗ 转换失败: {e}")
                results[key] = {
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
        for key, result in results.items():
            if result["status"] == "success":
                print(f"  ✓ {key}: {result['output_dir']}")
    if fail_count > 0:
        print("\n失败的数据集:")
        for key, result in results.items():
            if result["status"] == "failed":
                print(f"  ✗ {key}: {result['error']}")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('YOLO格式转COCO格式（用于seg目录）')
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='数据集目录路径（如 dataset/blur_seg），如果为None则批量转换所有')
    parser.add_argument('--image_type', type=str, default='png', choices=['png', 'rgb'],
                        help='图像类型：png 或 rgb')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认：数据集目录/COCO_format_{image_type}）')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例（默认0.8）')
    args = parser.parse_args()
    if args.dataset_dir:
        converter = SegYOLO2COCOConverter(args.dataset_dir, args.image_type, args.output_dir)
        converter.convert(args.train_ratio)
    else:
        convert_seg_datasets()
