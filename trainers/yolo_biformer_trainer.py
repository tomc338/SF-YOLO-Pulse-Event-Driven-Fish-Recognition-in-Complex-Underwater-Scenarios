import sys
import time
import torch
from pathlib import Path
from typing import Dict, Optional, Any
import warnings
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
ultralytics_src = project_root / "ultralytics"
if ultralytics_src.exists() and (ultralytics_src / "ultralytics").exists():
    if str(ultralytics_src) not in sys.path:
        sys.path.insert(0, str(ultralytics_src))
    print(f"[BiFormer] 使用源码版本的 ultralytics: {ultralytics_src}")
from trainers.yolo_trainer import YOLOTrainer
from models.biformer_neck import create_biformer_neck
warnings.filterwarnings("ignore")
EXTENDED_SUPPORTED_MODELS = {
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolo11m": "yolo11m.pt",
    "yolo11l": "yolo11l.pt",
    "yolo11x": "yolo11x.pt",
    "yolo26n": "yolo26n.pt",
    "yolo26s": "yolo26s.pt",
    "yolo26m": "yolo26m.pt",
    "yolo26l": "yolo26l.pt",
    "yolo26x": "yolo26x.pt",
    "yolov10n": "yolov10n.pt",
    "yolov10s": "yolov10s.pt",
    "yolov10m": "yolov10m.pt",
    "yolov10l": "yolov10l.pt",
    "yolov10x": "yolov10x.pt",
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    "yolov8x": "yolov8x.pt",
}
class YOLOBiFormerTrainer(YOLOTrainer):
    def __init__(self,
                 model_name: str = "yolo11n",
                 data_yaml: str = None,
                 project_name: str = "fish_detection",
                 experiment_name: str = None,
                 device: str = "auto",
                 pretrained: bool = True,
                 use_biformer_neck: bool = True,
                 biformer_config: Dict = None):
        if experiment_name is None:
            pretrained_tag = "pretrained" if pretrained else "from_scratch"
            experiment_name = f"{model_name}_biformer_{pretrained_tag}"
        super().__init__(
            model_name=model_name,
            data_yaml=data_yaml,
            project_name=project_name,
            experiment_name=experiment_name,
            device=device,
            pretrained=pretrained
        )
        self.use_biformer_neck = use_biformer_neck
        self.biformer_config = biformer_config or {}
        if use_biformer_neck:
            print("=" * 60)
            print("✓ 使用 BiFormer Neck")
            print("=" * 60)
            print(f"已集成到 ultralytics，将使用 {self.model_name}-biformer.yaml 配置")
            print("=" * 60)
    def load_model(self, pretrained: bool = None):
        if pretrained is None:
            pretrained = self.pretrained
        if 'ultralytics' in sys.modules:
            import importlib
            modules_to_reload = [
                'ultralytics.nn.modules.biformer',
                'ultralytics.nn.modules',
                'ultralytics.nn.tasks',
                'ultralytics.nn',
                'ultralytics.models.yolo.model',
                'ultralytics.models.yolo',
                'ultralytics.models',
                'ultralytics.engine.model',
                'ultralytics.engine',
                'ultralytics',
            ]
            for mod_name in modules_to_reload:
                if mod_name in sys.modules:
                    del sys.modules[mod_name]
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("请安装ultralytics: pip install ultralytics")
        if self.use_biformer_neck:
            base_name = self.model_name.replace("-biformer", "")
            if "yolo26" in base_name:
                yaml_dir = project_root / "ultralytics" / "ultralytics" / "cfg" / "models" / "26"
            elif "yolo11" in base_name:
                yaml_dir = project_root / "ultralytics" / "ultralytics" / "cfg" / "models" / "11"
            else:
                yaml_dir = project_root / "ultralytics" / "ultralytics" / "cfg" / "models" / "11"
            yaml_file = yaml_dir / f"{base_name}-biformer.yaml"
            if not yaml_file.exists():
                raise FileNotFoundError(
                    f"BiFormer YAML 配置文件不存在: {yaml_file}\n"
                    f"请确保已创建 {base_name}-biformer.yaml 配置文件"
                )
            print(f"加载模型 (使用 BiFormer Neck): {yaml_file.name}")
            print(f"   配置文件路径: {yaml_file}")
            if not pretrained:
                print("   从头训练（不使用预训练权重）")
                self.model = YOLO(str(yaml_file))
            else:
                try:
                    weights = EXTENDED_SUPPORTED_MODELS.get(base_name, f"{base_name}.pt")
                    print(f"   尝试加载预训练权重: {weights}")
                    print("   ⚠️  注意：BiFormer Neck 部分将随机初始化")
                    self.model = YOLO(str(yaml_file))
                    try:
                        self.model.load(weights, strict=False)
                        print(f"   ✓ 成功加载预训练权重（部分兼容）")
                    except Exception as e:
                        print(f"   ⚠️  预训练权重加载失败，将从头训练: {e}")
                except Exception as e:
                    print(f"   ⚠️  权重加载失败，将从头训练: {e}")
                    self.model = YOLO(str(yaml_file))
        else:
            return super().load_model(pretrained)
        self._log_model_info()
        return self.model
    def _replace_neck_if_possible(self):
        if not self.use_biformer_neck or self.biformer_neck is None:
            return False
        try:
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'neck'):
                print("尝试替换 YOLO neck...")
                print("⚠️  无法自动替换，需要手动修改 ultralytics 源码")
                return False
            else:
                print("⚠️  无法找到 YOLO neck 模块")
                return False
        except Exception as e:
            print(f"⚠️  替换 neck 失败: {e}")
            return False
def train_yolo_biformer(model_name: str = "yolo11n",
                        data_yaml: str = None,
                        epochs: int = 100,
                        batch_size: int = 16,
                        imgsz: int = 640,
                        device: str = "auto",
                        pretrained: bool = True,
                        **kwargs) -> Dict:
    trainer = YOLOBiFormerTrainer(
        model_name=model_name,
        data_yaml=data_yaml,
        device=device,
        pretrained=pretrained,
        **kwargs
    )
    trainer.load_model()
    results = trainer.train(
        epochs=epochs,
        batch_size=batch_size,
        imgsz=imgsz,
        **kwargs
    )
    trainer.save_results()
    return results
if __name__ == "__main__":
    print("YOLO with BiFormer Neck 训练器测试")
    print("=" * 60)
    print("注意：此训练器需要自定义 YOLO 架构才能实际使用 BiFormer Neck")
    print("当前实现提供框架，实际使用需要修改 ultralytics 源码或使用自定义实现")
    print("=" * 60)
