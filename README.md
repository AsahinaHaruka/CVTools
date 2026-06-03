<div align="right">
  <strong><a href="#chinese">简体中文</a> | <a href="#english">English</a></strong>
</div>

---

<a id="chinese"></a>
# CVKit

这是一个用于常见计算机视觉任务的工具包。

## 文件与模块说明

所有核心代码均位于 `src/cvkit` 包内：

* **`cvkit/inference`**: ONNX 模型推理模块。
    * `base_inference.py`: 推理基类。
    * `yolo_inference.py`: YOLO 模型推理。
    * `sam3_inference.py`: SAM3 模型推理。
    * `tokenizer.py`: SAM3 的文本编码器。
    * `image_processor.py`: 图像处理器。
* **`cvkit/utils`**: 通用工具模块。
    * `perspective_transformation.py`: 图像透视变换工具。
    * `logger.py`: 日志工具.
    * `data_define.py`: 通用数据结构定义。
    * `rtsp_video.py`: RTSP 视频拉流工具。
* **命令行工具 (CLI)**:
    * `cvkit-color-to-gray`: 将彩色图像转换为灰度图。
    * `cvkit-data-split`: 随机拆分训练集和测试集（支持 YOLO 格式数据）。
    * `cvkit-random-selection`: 从目录中随机选择并复制指定数量的文件。
    * `cvkit-show-yolo-obj`: 可视化 YOLO 目标检测的 ONNX 模型推理标注结果。
    * `cvkit-show-yolo-seg`: 可视化 YOLO 实例分割的 ONNX 模型推理标注结果。
    * `cvkit-video2image`: 将视频文件转换为图像序列（支持选点和透视变换）。

---

## 安装说明

本工具包已支持通过 pip / uv 安装。您可以根据硬件环境选择不同的推理加速后端：

### 1. 基础安装（仅使用基础图像处理，不使用 ONNX 推理）
```bash
pip install cvkit-python
# 或者使用 uv
uv pip install cvkit-python
```

### 2. 按需安装硬件加速后端
* **使用 CPU 推理**
  ```bash
  pip install "cvkit-python[onnx-cpu]"
  ```
* **使用 GPU 推理（支持 TensorRT / CUDA）**
  ```bash
  pip install "cvkit-python[onnx-gpu]"
  ```
* **使用 SAM3 推理（安装额外文本编码依赖）**
  ```bash
  pip install "cvkit-python[onnx-gpu,sam]"
  ```

---

## 使用方法

### 作为命令行工具使用

安装后，您可以在终端中直接使用以下命令：

```bash
# 1. 批量图像转灰度
cvkit-color-to-gray -i ./input_dir -o ./output_dir --recursive

# 2. 随机拆分 YOLO 格式数据集
cvkit-data-split -i ./dataset_dir -o ./split_out --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1

# 3. 随机抽取指定数量图片
cvkit-random-selection -i ./src_dir -o ./dst_dir -n 100 --model fixed

# 4. 可视化 YOLO 目标检测模型推理结果
cvkit-show-yolo-obj --model ./yolov8n.onnx -i ./images -o ./output_results

# 5. 可视化 YOLO 实例分割模型推理结果
cvkit-show-yolo-seg --model ./yolov8n-seg.onnx -i ./images -o ./output_results --conf 0.5

# 6. 视频提取帧（支持透视选点）
cvkit-video2image -i ./video.mp4 -o ./frames --perspective
```

### 作为 Python 库导入

```python
from cvkit.utils.rtsp_video import Video
from cvkit.inference.yolo_inference import YoloObjInference
from cvkit.utils.logger import LoggerBuilder

# 初始化日志
logger = LoggerBuilder().get_logger("my_app")

# 使用 RTSP 视频拉流
video = Video(ip="192.168.1.100", stream_suffix="live")
```

---
<a id="english"></a>
#  CVKit

A computer vision toolkit featuring modular design.

## Modules and File Descriptions

All core code resides inside the `src/cvkit` package:

* **`cvkit/inference`**: Module for ONNX model inference.
    * `base_inference.py`: Base class for inference.
    * `yolo_inference.py`: Inference for YOLO models.
    * `sam3_inference.py`: Inference for SAM3 models.
    * `tokenizer.py`: Text encoder for SAM3.
    * `image_processor.py`: Image processor.
* **`cvkit/utils`**: General utility module.
    * `perspective_transformation.py`: A class for performing perspective transformation.
    * `logger.py`: Logging utility.
    * `data_define.py`: Common data structure definitions.
    * `rtsp_video.py`: RTSP video stream ingester.
* **Command Line Interfaces (CLI)**:
    * `cvkit-color-to-gray`: Converts color images to grayscale.
    * `cvkit-data-split`: Splits dataset into train, val, and test subsets.
    * `cvkit-random-selection`: Randomly selects a portion of files.
    * `cvkit-show-yolo-obj`: Visualizes YOLO object detection labels.
    * `cvkit-show-yolo-seg`: Visualizes YOLO instance segmentation labels.
    * `cvkit-video2image`: Converts a video file into an image sequence with perspective options.

---

## Installation

You can install CVTools using pip or uv, optionally installing target computing backends for hardware acceleration:

### 1. Basic Installation (No ONNX inference)
```bash
pip install cvkit-python
# or with uv
uv pip install cvkit-python
```

### 2. Accelerated Backends Installation
* **For CPU Inference**
  ```bash
  pip install "cvkit-python[onnx-cpu]"
  ```
* **For GPU Inference (CUDA/TensorRT)**
  ```bash
  pip install "cvkit-python[onnx-gpu]"
  ```
* **For SAM3 Inference**
  ```bash
  pip install "cvkit-python[onnx-gpu,sam]"
  ```

---

## Usage

### Command Line Interface

You can call CVTools scripts directly from the terminal after installation:

```bash
# 1. Grayscale Conversion
cvkit-color-to-gray -i ./input_dir -o ./output_dir --recursive

# 2. Dataset Splitting
cvkit-data-split -i ./dataset_dir -o ./split_out --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1

# 3. Random Selection
cvkit-random-selection -i ./src_dir -o ./dst_dir -n 100 --model fixed

# 4. YOLO Object Detection Inference & Visualization
cvkit-show-yolo-obj --model ./yolov8n.onnx -i ./images -o ./output_results

# 5. YOLO Instance Segmentation Inference & Visualization
cvkit-show-yolo-seg --model ./yolov8n-seg.onnx -i ./images -o ./output_results --conf 0.5

# 6. Video Frame Extraction with Perspective Selection
cvkit-video2image -i ./video.mp4 -o ./frames --perspective
```

### Python Library Import

```python
from cvkit.utils.rtsp_video import Video
from cvkit.inference.yolo_inference import YoloObjInference
from cvkit.utils.logger import LoggerBuilder

# Initialize logger
logger = LoggerBuilder().get_logger("my_app")

# RTSP Video Stream ingestion
video = Video(ip="192.168.1.100", stream_suffix="live")
```