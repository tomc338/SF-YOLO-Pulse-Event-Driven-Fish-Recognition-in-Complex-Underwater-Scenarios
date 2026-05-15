# SF-YOLO: Pulse Event-Driven Fish Recognition 🐟

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

本项目为国家级大学生创新训练项目——《脉冲事件驱动的复杂水下场景鱼体识别系统》的官方开源仓库。

## 🌊 项目背景 (Background)

传统水下视觉检测在低照度、高模糊及悬浮物遮挡等退化环境下极易失效。本项目提出了一种**仿真脉冲事件驱动**的解决方案：
1. **克服硬件限制**：通过 ODG 轨迹驱动算法仿真生成纯净脉冲信号，解决了事件相机水下部署成本高的问题。
2. **提升检测鲁棒性**：利用脉冲图像的高频轮廓提取能力，有效滤除水下背景噪声。

## ✨ 核心技术 (Key Features)

### 1. 脉冲信号仿真与重构 (Event Simulation)
* **ODG 轨迹驱动**：通过预设运动轨迹激发静态图像，生成高保真脉冲流。
* **边缘补偿算法**：结合时域积分技术，在重构的二维脉冲图像中精准保留目标边缘，滤除水下冗余背景。

### 2. SF-YOLO 检测算法
* **骨干网络**：以 **YOLOv26** 为基座，针对脉冲图像特性进行深度优化。
* **SEAM 注意力机制**：引入多尺度注意力机制，增强模型对细微特征的感知能力，解决脉冲图像细节单调导致的特征混叠问题。
* **CSMM 模块**：结合深度可分离卷积与残差连接，在保证轻量化的同时提升特征提取效率。

## 📊 实验结果 (Results)

在复杂水下场景下的实验结果显示，SF-YOLO 表现出极强的鲁棒性：
* **模糊场景**：mAP@0.5 提升约 **36%**。
* **昏暗场景**：mAP@0.5 提升约 **46%**。

*对比实验涵盖了原生 YOLO、Faster R-CNN 等基准模型。*

## 🚀 快速开始 (Quick Start)

### 环境安装
```bash
git clone [https://github.com/YourUsername/SF-YOLO.git](https://github.com/YourUsername/SF-YOLO.git)
cd SF-YOLO
pip install -r requirements.txt
