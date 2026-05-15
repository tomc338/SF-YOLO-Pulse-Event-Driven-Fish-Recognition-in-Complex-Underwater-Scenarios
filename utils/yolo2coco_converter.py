
import json
import os
import shutil
import time
from pathlib import Path
import cv2
import yaml
from tqdm import tqdm
class YOLO2COCOConverter:
    def __init__(self, yaml_path, output_dir=None):
        self.yaml_path = Path(yaml_path)
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"YAML文件不存在: {yaml_path}")
        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            self.data_cfg = yaml.safe_load(f)
        self.root_dir = self.yaml_path.parent.parent
        self.root_data_dir = Path(self.data_cfg.get('path'))
        if not self.root_data_dir.is_absolute():
            self.root_data_dir = self.root_dir / self.root_data_dir
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.root_data_dir / "COCO_format"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_path = self._get_data_dir('train')
        self.val_path = self._get_data_dir('val')
        self.nc = self.data_cfg['nc']
        if 'names' in self.data_cfg:
            self.names = self.data_cfg.get('names')
        else:
            self.names = [f'class{i}' for i in range(self.nc)]
        assert len(self.names) == self.nc,\
            f'{len(self.names)} names found for nc={self.nc} dataset'
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
        self.categories = []
        self._get_category()
        self.annotation_id = 1
        cur_year = time.strftime('%Y', time.localtime(time.time()))
        self.info = {
            'year': int(cur_year),
            'version': '1.0',
            'description': 'For object detection',
            'date_created': cur_year,
        }
        self.licenses = [{
            'id': 1,
            'name': 'Apache License v2.0',
            'url': 'https://github.com/RapidAI/YOLO2COCO/LICENSE',
        }]
    def _get_data_dir(self, mode):
        data_dir = self.data_cfg.get(mode)
        if data_dir:
            if isinstance(data_dir, str):
                full_path = [str(self.root_data_dir / data_dir)]
            elif isinstance(data_dir, list):
                full_path = [str(self.root_data_dir / one_dir) for one_dir in data_dir]
            else:
                raise TypeError(f'{data_dir} is not str or list.')
        else:
            raise ValueError(f'{mode} dir is not in the yaml.')
        return full_path
    def _get_category(self):
        for i, category in enumerate(self.names, start=1):
            self.categories.append({
                'supercategory': category,
                'id': i,
                'name': category,
            })
    def get_files(self, path):
        IMG_FORMATS = ['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp']
        f = []
        for p in path:
            p = Path(p)
            if p.is_dir():
                f += list(p.rglob('*.*'))
            elif p.is_file():
                with open(p) as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [x.replace('./', parent) if x.startswith('./') else x for x in t]
            else:
                raise Exception(f'{p} does not exist')
        im_files = []
        for x in f:
            x_path = Path(x)
            if x_path.suffix.lower()[1:] in IMG_FORMATS:
                im_files.append(str(x_path))
        return sorted(im_files)
    def gen_dataset(self, img_paths, target_img_path, target_json, mode):
        images = []
        annotations = []
        sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep
        img_id = 0
        skipped_count = 0
        for img_path in tqdm(img_paths, desc=f"转换{mode}集"):
            label_path = sb.join(img_path.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt'
            img_path = Path(img_path)
            if not img_path.exists():
                skipped_count += 1
                continue
            label_path = Path(label_path)
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
        print(f"[OK] {mode}集转换完成: {len(images)}张图像, {len(annotations)}个标注")
        if skipped_count > 0:
            print(f"  (跳过{skipped_count}个无效/无标注图像)")
        return len(images), len(annotations)
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
    def convert(self):
        print(f"开始转换YOLO格式到COCO格式...")
        print(f"数据集路径: {self.root_data_dir}")
        print(f"输出路径: {self.output_dir}")
        print(f"类别数: {self.nc}, 类别: {self.names}")
        print()
        train_files = self.get_files(self.train_path)
        val_files = self.get_files(self.val_path)
        print(f"训练集: {len(train_files)}张图像")
        print(f"验证集: {len(val_files)}张图像")
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
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('YOLO格式转COCO格式（保留原文件名）')
    parser.add_argument('--yaml_path', type=str, required=True,
                        help='数据集YAML配置文件路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认：数据集目录/COCO_format）')
    args = parser.parse_args()
    converter = YOLO2COCOConverter(args.yaml_path, args.output_dir)
    converter.convert()
