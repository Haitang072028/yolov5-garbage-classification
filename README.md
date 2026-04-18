
# 基于YOLOv5的垃圾分类算法研究

本项目以YOLOv5目标检测框架为核心，针对垃圾分类场景中目标尺度差异大、遮挡频繁等问题，引入 **CBAM注意力机制** 和 **Focal-CIoU损失函数**，显著提升了检测精度与鲁棒性，并基于PyQt5开发了图形化检测系统。

**改进后模型在测试集上达到：**
- mAP@0.5: **91.9%**
- mAP@0.5:0.95: **63.9%**

---

## 🚀 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/Haitang072028/yolov5-garbage-classification.git
cd yolov5-garbage-classification
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```
主要依赖：Python 3.8，PyTorch 1.13.1，torchvision 0.14.1，opencv-python，pyqt5，numpy，matplotlib

### 3. 准备数据集
将数据集按以下结构放入 `large_rubbish_class/`：
```
large_rubbish_class/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```
修改 `data/data1.yaml` 配置文件，指定路径和类别数（4类）。

### 4. 训练模型
使用默认配置训练（数据集路径、模型配置文件等已预设）：
```bash
python train.py --epochs 300 --batch-size 4 --device 0
```
如需修改数据集或模型配置，可直接在命令行添加参数，例如：
```bash
python train.py --data your_data.yaml --cfg your_model.yaml --weights your_weights.pt --epochs 300 --batch-size 4
```

### 5. 图片/视频检测
使用预训练权重或自己训练好的权重：
```bash
python detect.py --weights yolov5s.pt --source large_rubbish_class/images/test --project runs/detect/
```
如果想使用训练好的最佳权重，将 `--weights` 改为 `runs/train/exp/weights/best.pt`。

### 6. 启动 PyQt5 图形界面
```bash
python window.py
```

---

## 📁 数据集

- **来源**：ModelScope 垃圾分类数据集（筛选后共 8370张有效图片）
- **类别**：可回收垃圾、厨余垃圾、有害垃圾、其他垃圾
- **划分**：训练集 6696，验证集 837，测试集 837（8:1:1）

数据集已在仓库中的 `large_rubbish_class/` 目录下。
数据集地址：[https://github.com/Haitang072028/yolov5-garbage-classification/tree/master/large_rubbish_class](https://github.com/Haitang072028/yolov5-garbage-classification/tree/master/large_rubbish_class)

---

## 🧠 模型改进与结果

基于 YOLOv5s，进行了两项优化：

| 改进点 | 方法 | 作用 |
|--------|------|------|
| 特征增强 | **CBAM**（通道+空间双注意力） | 强化关键特征，抑制背景干扰 |
| 损失函数 | **Focal-CIoU** | 缓解类别不平衡，提升定位精度 |

**消融实验对比：**

| 模型 | mAP@0.5 | mAP@0.5:0.95 |
|------|---------|--------------|
| YOLOv5s（基线） | 86.5% | 58.2% |
| +CBAM | 89.7% | 61.3% |
| +Focal-CIoU | 90.1% | 62.0% |
| **CBAM+Focal-CIoU（本文）** | **91.9%** | **63.9%** |

### 鲁棒性测试结论
- **亮度波动**：曝光率在 -4% ~ +8% 范围内，准确率 >70%
- **局部遮挡**：左右部分遮挡时仍能准确分类
- **噪声干扰**：0~30% 高斯噪声下，准确率 >90%

---

## 🖼️ 检测效果示例

| 原图 | 检测结果 |
|------|----------|
| ![原图](https://cdn.jsdelivr.net/gh/Haitang072028/yolov5-garbage-classification@master/assets/p11.jpg) | ![检测结果](https://cdn.jsdelivr.net/gh/Haitang072028/yolov5-garbage-classification@master/assets/p10.jpg) |



---

## 💻 PyQt5 检测系统

| 界面展示 | 检测图片 |
|----------|----------|
| ![界面1](https://cdn.jsdelivr.net/gh/Haitang072028/yolov5-garbage-classification@master/assets/system1.jpg) | ![检测](https://cdn.jsdelivr.net/gh/Haitang072028/yolov5-garbage-classification@master/assets/system3.jpg) |
| ![界面2](https://cdn.jsdelivr.net/gh/Haitang072028/yolov5-garbage-classification@master/assets/system2.jpg) | |

支持图片上传、实时视频流分析及多格式文件处理。



