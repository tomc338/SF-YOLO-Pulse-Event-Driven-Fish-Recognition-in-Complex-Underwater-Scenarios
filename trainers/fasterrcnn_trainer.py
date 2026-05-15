
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict
import yaml
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils.metrics_logger import MetricsLogger
class FasterRCNNTrainer:
    SUPPORTED_BACKBONES = {
        "mobilenet_v3_large_fpn": {"params": 19, "flops": 4.5},
        "resnet50_fpn": {"params": 41, "flops": 134},
        "resnet50_fpn_v2": {"params": 43, "flops": 280},
    }
    def __init__(self, model_name="faster_rcnn", backbone="resnet50_fpn",
                 data_yaml=None, project_name="fish_detection",
                 experiment_name=None, device="auto", pretrained=False):
        self.model_name = model_name.lower()
        self.backbone = backbone.lower()
        self.data_yaml = data_yaml
        self.project_name = project_name
        self.pretrained = pretrained
        self.num_classes = 2
        self.class_names = ['fish']
        self.coco_path = None
        if data_yaml:
            self._parse_data_yaml(data_yaml)
        pretrained_tag = "pretrained" if pretrained else "scratch"
        backbone_short = backbone.replace("_fpn", "").replace("_v3_large", "")
        self.experiment_name = experiment_name or f"fasterrcnn_{backbone_short}_{pretrained_tag}"
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu"
        self.model = None
        self.save_dir = project_root / "results" / self.project_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = Path(data_yaml).stem if data_yaml else "unknown"
        self.logger = MetricsLogger(
            experiment_name=self.experiment_name,
            model_name=f"FasterRCNN-{backbone_short}",
            dataset_name=dataset_name,
            save_dir=str(self.save_dir)
        )
    def _parse_data_yaml(self, data_yaml):
        with open(data_yaml, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        nc = config.get('nc', 1)
        self.num_classes = nc + 1
        self.class_names = config.get('names', ['fish'])
        base_path = Path(config.get('path', ''))
        if not base_path.is_absolute():
            base_path = project_root / base_path
        coco_format_path = base_path / "COCO_format"
        if coco_format_path.exists():
            self.coco_path = str(coco_format_path)
            print(f"Found COCO format data: {self.coco_path}")
        else:
            self.coco_path = None
            print("COCO_format not found. Run: python convert_all_to_coco.py")
    def load_model(self, pretrained=None):
        if pretrained is None:
            pretrained = self.pretrained
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2,
            fasterrcnn_mobilenet_v3_large_fpn
        )
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        print(f"Loading Faster R-CNN (backbone: {self.backbone})")
        print(f"Mode: {'pretrained' if pretrained else 'from scratch'}")
        weights = None
        if pretrained:
            if "v2" in self.backbone:
                from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
                weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            elif "mobilenet" in self.backbone:
                from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights
                weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            else:
                from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
                weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        if "v2" in self.backbone:
            self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        elif "mobilenet" in self.backbone:
            self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
        else:
            self.model = fasterrcnn_resnet50_fpn(weights=weights)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        self.model.roi_heads.score_thresh = 0.01
        if not pretrained:
            self._reinit_weights()
        self.model.to(self.device)
        self._log_model_info()
        return self.model
    def _reinit_weights(self):
        print("Reinitializing all weights...")
        def init_weights(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
        self.model.apply(init_weights)
        print("✓ All weights reinitialized with stable initialization")
    def _log_model_info(self):
        if self.model is None:
            return
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        backbone_info = self.SUPPORTED_BACKBONES.get(self.backbone, {"params": 41, "flops": 134})
        flops = backbone_info["flops"]
        model_size = total_params * 4 / (1024 * 1024)
        self.logger.set_model_info(params=total_params, flops=flops,
                                   model_size_mb=model_size, input_size=(640, 640))
        print(f"Parameters: {total_params/1e6:.2f}M")
        print(f"Trainable: {trainable_params/1e6:.2f}M")
        print(f"FLOPs: {flops}G")
        print(f"Model size: {model_size:.2f}MB")
    def train(self, epochs=100, batch_size=2, imgsz=640, lr0=0.005,
              momentum=0.9, weight_decay=0.0005, warmup_epochs=5,
              patience=50, save_period=10, workers=0, **kwargs):
        if self.model is None:
            self.load_model()
        if self.coco_path is None:
            raise FileNotFoundError("COCO format data required!")
        config = {
            "model": f"FasterRCNN-{self.backbone}", "backbone": self.backbone,
            "pretrained": self.pretrained, "epochs": epochs, "batch_size": batch_size,
            "lr0": lr0, "momentum": momentum, "weight_decay": weight_decay,
            "device": self.device, "num_classes": self.num_classes,
        }
        self.logger.set_config(config)
        print(f"\n{'='*60}")
        print(f"Training: Faster R-CNN ({self.backbone})")
        print(f"Dataset: {self.data_yaml}")
        print(f"Device: {self.device}, Epochs: {epochs}, Batch: {batch_size}")
        print(f"From scratch: {'No' if self.pretrained else 'Yes'}")
        print(f"{'='*60}\n")
        start_time = time.time()
        best_map = 0.0
        from torchvision.datasets import CocoDetection
        import torchvision.transforms as T
        coco_path = Path(self.coco_path)
        train_img = coco_path / "train2017"
        val_img = coco_path / "val2017"
        train_ann = coco_path / "annotations" / "instances_train2017.json"
        val_ann = coco_path / "annotations" / "instances_val2017.json"
        class COCODataset(CocoDetection):
            def __getitem__(self, idx):
                img, target = super().__getitem__(idx)
                boxes, labels, areas, iscrowd = [], [], [], []
                real_image_id = self.ids[idx]
                for obj in target:
                    bbox = obj['bbox']
                    boxes.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
                    labels.append(obj['category_id'])
                    areas.append(obj['area'])
                    iscrowd.append(obj.get('iscrowd', 0))
                td = {'image_id': torch.tensor([real_image_id])}
                if boxes:
                    td['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
                    td['labels'] = torch.as_tensor(labels, dtype=torch.int64)
                    td['area'] = torch.as_tensor(areas, dtype=torch.float32)
                    td['iscrowd'] = torch.as_tensor(iscrowd, dtype=torch.int64)
                else:
                    td['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                    td['labels'] = torch.zeros((0,), dtype=torch.int64)
                    td['area'] = torch.zeros((0,), dtype=torch.float32)
                    td['iscrowd'] = torch.zeros((0,), dtype=torch.int64)
                return img, td
        transform = T.Compose([T.ToTensor()])
        dataset_train = COCODataset(str(train_img), str(train_ann), transform=transform)
        dataset_val = COCODataset(str(val_img), str(val_ann), transform=transform)
        print(f"Train: {len(dataset_train)} images, Val: {len(dataset_val)} images")
        def collate_fn(batch):
            return tuple(zip(*batch))
        loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True,
            num_workers=workers, collate_fn=collate_fn,
            pin_memory=False
        )
        loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=batch_size, shuffle=False,
            num_workers=workers, collate_fn=collate_fn,
            pin_memory=False
        )
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=lr0, momentum=momentum, weight_decay=weight_decay)
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return 0.01 + 0.99 * epoch / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * progress))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        output_dir = self.save_dir / "runs" / self.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            loss_cls, loss_box, loss_obj, loss_rpn = 0.0, 0.0, 0.0, 0.0
            num_batches = 0
            for images, targets in loader_train:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                valid_idx = [i for i, t in enumerate(targets) if len(t['boxes']) > 0]
                if not valid_idx:
                    continue
                images = [images[i] for i in valid_idx]
                targets = [targets[i] for i in valid_idx]
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                if not torch.isfinite(losses):
                    print(f"⚠️  警告：Loss变成{losses.item()}，跳过此batch")
                    continue
                optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                optimizer.step()
                epoch_loss += losses.item()
                loss_cls += loss_dict.get('loss_classifier', torch.tensor(0)).item()
                loss_box += loss_dict.get('loss_box_reg', torch.tensor(0)).item()
                loss_obj += loss_dict.get('loss_objectness', torch.tensor(0)).item()
                loss_rpn += loss_dict.get('loss_rpn_box_reg', torch.tensor(0)).item()
                num_batches += 1
            lr_scheduler.step()
            avg_loss = epoch_loss / max(num_batches, 1)
            self.logger.log_train_loss(
                epoch=epoch+1, loss=avg_loss,
                cls_loss=loss_cls/max(num_batches,1),
                box_loss=loss_box/max(num_batches,1),
                objectness_loss=loss_obj/max(num_batches,1),
                rpn_box_loss=loss_rpn/max(num_batches,1)
            )
            if (epoch + 1) % save_period == 0 or epoch == epochs - 1:
                val_metrics = self._evaluate(loader_val)
                self.logger.log_epoch_metrics(
                    epoch=epoch+1, precision=val_metrics['precision'],
                    recall=val_metrics['recall'], map50=val_metrics['mAP@0.5'],
                    map50_95=val_metrics['mAP@0.5:0.95']
                )
                if val_metrics['mAP@0.5'] > best_map:
                    best_map = val_metrics['mAP@0.5']
                    torch.save({'model': self.model.state_dict(), 'epoch': epoch,
                               'best_map': best_map}, output_dir / 'best.pth')
                    print(f"  Saved best model (mAP@0.5: {best_map:.4f})")
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        training_time = time.time() - start_time
        print(f"\nTraining complete! Best mAP@0.5: {best_map:.4f}")
        print(f"Time: {training_time/3600:.2f}h")
        return {"model_name": f"FasterRCNN-{self.backbone}",
                "training_time": training_time, "best_map": best_map,
                "best_metrics": self.logger.get_best_metrics()}
    def _evaluate(self, data_loader):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        import json
        import tempfile
        self.model.eval()
        coco_results = []
        with torch.no_grad():
            for images, targets in data_loader:
                images = [img.to(self.device) for img in images]
                outputs = self.model(images)
                for target, output in zip(targets, outputs):
                    image_id = int(target['image_id'][0])
                    boxes = output['boxes'].cpu()
                    scores = output['scores'].cpu()
                    labels = output['labels'].cpu()
                    for box, score, label in zip(boxes, scores, labels):
                        x1, y1, x2, y2 = box.tolist()
                        coco_results.append({
                            'image_id': image_id,
                            'category_id': int(label),
                            'bbox': [x1, y1, x2 - x1, y2 - y1],
                            'score': float(score)
                        })
        print(f"  调试：收集到 {len(coco_results)} 个检测框")
        if len(coco_results) > 0:
            scores = [r['score'] for r in coco_results]
            cats = [r['category_id'] for r in coco_results]
            print(f"  置信度范围: {min(scores):.4f} ~ {max(scores):.4f}")
            print(f"  预测类别: {set(cats)} (期望: {1})")
            print(f"  前3个预测示例:")
            for i, r in enumerate(coco_results[:3]):
                print(f"    [{i}] img={r['image_id']}, cat={r['category_id']}, score={r['score']:.3f}, bbox={[f'{x:.1f}' for x in r['bbox']]}")
        if len(coco_results) == 0:
            print("  ⚠️  警告：模型没有输出任何检测结果")
            return {'precision': 0.0, 'recall': 0.0, 'mAP@0.5': 0.0, 'mAP@0.5:0.95': 0.0}
        try:
            coco_path = Path(self.coco_path)
            ann_file = coco_path / "annotations" / "instances_val2017.json"
            coco_gt = COCO(str(ann_file))
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(coco_results, f)
                temp_file = f.name
            coco_dt = coco_gt.loadRes(temp_file)
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            eval_imgs = coco_eval.evalImgs
            matched = sum(1 for e in eval_imgs if e is not None and len(e['dtMatches'][0]) > 0)
            total_imgs = sum(1 for e in eval_imgs if e is not None)
            print(f"  调试：{matched}/{total_imgs} 张图有检测框匹配到GT (IoU>0.5)")
            coco_eval.summarize()
            import os
            os.unlink(temp_file)
            stats = coco_eval.stats
            return {
                'precision': stats[0],
                'recall': stats[8] if len(stats) > 8 else 0,
                'mAP@0.5': stats[1],
                'mAP@0.5:0.95': stats[0]
            }
        except Exception as e:
            print(f"  ⚠️  COCO评估失败: {e}")
            return self._simple_evaluate(data_loader)
    def _simple_evaluate(self, data_loader):
        self.model.eval()
        total_tp, total_fp, total_fn = 0, 0, 0
        with torch.no_grad():
            for images, targets in data_loader:
                images = [img.to(self.device) for img in images]
                outputs = self.model(images)
                for output, target in zip(outputs, targets):
                    pred_boxes = output['boxes'].cpu()
                    pred_scores = output['scores'].cpu()
                    gt_boxes = target['boxes']
                    keep = pred_scores > 0.05
                    pred_boxes = pred_boxes[keep]
                    if len(gt_boxes) == 0:
                        total_fp += len(pred_boxes)
                        continue
                    if len(pred_boxes) == 0:
                        total_fn += len(gt_boxes)
                        continue
                    matched = 0
                    for pb in pred_boxes:
                        for gb in gt_boxes:
                            iou = self._iou(pb, gb)
                            if iou > 0.5:
                                matched += 1
                                break
                    total_tp += matched
                    total_fp += len(pred_boxes) - matched
                    total_fn += len(gt_boxes) - matched
        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return {'precision': precision, 'recall': recall,
                'mAP@0.5': f1,
                'mAP@0.5:0.95': f1 * 0.8}
    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        return inter / (area1 + area2 - inter + 1e-8)
    def benchmark_inference(self, imgsz=640, warmup=10, runs=100):
        if self.model is None:
            self.load_model()
        self.model.eval()
        dummy = [torch.rand(3, imgsz, imgsz).to(self.device)]
        print(f"Inference benchmark (warmup={warmup}, runs={runs})...")
        with torch.no_grad():
            for _ in range(warmup):
                self.model(dummy)
        if self.device == "cuda":
            torch.cuda.synchronize()
        times = []
        with torch.no_grad():
            for _ in range(runs):
                start = time.perf_counter()
                self.model(dummy)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1000 / avg_time
        self.logger.set_inference_metrics(
            inference_time_ms=avg_time, fps=fps, latency_ms=avg_time,
            batch_size=1, input_size=(imgsz, imgsz), device=self.device)
        print(f"Inference: {avg_time:.2f}ms (+/-{std_time:.2f}ms)")
        print(f"FPS: {fps:.2f}, Latency: {avg_time:.2f}ms")
        return {"inference_time_ms": avg_time, "fps": fps, "latency_ms": avg_time}
    def get_results(self):
        return {"model_name": f"FasterRCNN-{self.backbone}",
                "experiment_name": self.experiment_name,
                "model_info": self.logger.model_info,
                "config": self.logger.config,
                "best_metrics": self.logger.get_best_metrics(),
                "inference_metrics": self.logger.inference_metrics,
                "save_dir": str(self.logger.exp_dir)}
    def save_results(self):
        excel_path, json_path = self.logger.save_all()
        print(f"\nResults saved:\n  Excel: {excel_path}\n  JSON: {json_path}")
        return excel_path, json_path
if __name__ == "__main__":
    print("Faster R-CNN Trainer Test")
    trainer = FasterRCNNTrainer(
        backbone="mobilenet_v3_large_fpn",
        data_yaml="configs/dataset_blur_png.yaml",
        pretrained=False
    )
    trainer.load_model()
    trainer.benchmark_inference(warmup=3, runs=10)
    trainer.save_results()
    print("Test complete!")
