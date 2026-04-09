<div align="right">
  <strong><a href="#chinese">简体中文</a> | <a href="#english">English</a></strong>
</div>

---

<a id="chinese"></a>
# CVTools

这是一个用于常见计算机视觉任务的脚本合集。

## 文件说明

* [inference](inference): ONNX模型推理模块。
    * [base_inference.py](inference/base_inference.py): 推理基类。
    * [yolo_inference.py](inference/yolo_inference.py): YOLO模型推理。
    * [sam3_inference.py](inference/sam3_inference.py): SAM3模型推理。
    * [tokenizer.py](inference/tokenizer.py): SAM3的文本编码器。
    * [image_processor.py](inference/image_processor.py): 图像处理器。
* [utils](utils): 通用工具模块。
    * [perspective_transformation.py](utils/perspective_transformation.py): 对图像进行透视变换的类。
    * [logger.py](utils/logger.py): 日志工具。
    * [data_define.py](utils/data_define.py): 通用数据结构定义。
* [video2image.py](video2image.py): 将视频文件转换为图像序列，包含选点和透视变换。
* [color_to_gray.py](color_to_gray.py): 将彩色图像转换为灰度图。
* [random_selection.py](random_selection.py): 从目录中随机选择一部分文件。
* [data_split.py](data_split.py): 随机拆分训练集和测试集。
* [show_yolo_obj_image.py](show_yolo_obj_image.py): 可视化YOLO目标检测的标注。
* [show_yolo_seg_image.py](show_yolo_seg_image.py): 可视化YOLO实例分割的标注。
* [rtsp_video.py](rtsp_video.py): RTSP拉流。

## 安装与使用

1.  **安装依赖**

    本仓库使用 uv 来管理依赖，请确保您已安装 uv。

    ```bash
    pip install uv
    ```

2. **按需安装项目依赖**

   根据您的需求，选择以下**其中一种**方式同步依赖：

    * **仅使用基础图像处理工具（不进行 ONNX 推理）**
        ```bash
        uv sync
        ```
    * **使用 CPU 进行模型推理**
        ```bash
        uv sync --extra onnx-cpu
        ```
    * **使用 GPU 进行模型推理**
        ```bash
        uv sync --extra onnx-gpu
        ```
    * **推理SAM模型**
      ```bash
      uv sync --extra onnx-gpu --extra sam
      ```
    * **安装所有功能与依赖**
      ```bash
      uv sync --all-extras
      ```

3. **运行脚本**

   使用 `uv run` 来执行脚本：
   ```bash
   uv run <script_name>.py
   ```

---
<a id="english"></a>
# CVTools

This is a collection of scripts for common computer vision tasks, featuring a modular design that allows for on-demand
installation of hardware acceleration backends.

## File Descriptions

* [inference](inference): Module for ONNX model inference.
    * [base_inference.py](inference/base_inference.py): Base class for inference.
    * [yolo_inference.py](inference/yolo_inference.py): Inference for YOLO models.
    * [sam3_inference.py](inference/sam3_inference.py): Inference for SAM3 models.
    * [tokenizer.py](inference/tokenizer.py): Text encoder for SAM3.
    * [image_processor.py](inference/image_processor.py): Image processor.
* [utils](utils): General utility module.
    * [perspective_transformation.py](utils/perspective_transformation.py): A class for performing perspective
      transformation on an image.
    * [logger.py](utils/logger.py): Logging utility.
    * [data_define.py](utils/data_define.py): Common data structure definitions.
* [video2image.py](video2image.py): Converts a video file into an image sequence, including point selection and
  perspective transformation.
* [color_to_gray.py](color_to_gray.py): Converts a color image to grayscale.
* [random_selection.py](random_selection.py): Randomly selects a portion of files from a directory.
* [data_split.py](data_split.py): Randomly splits a dataset into training and testing sets.
* [show_yolo_obj_image.py](show_yolo_obj_image.py): Visualizes YOLO object detection labels.
* [show_yolo_seg_image.py](show_yolo_seg_image.py): Visualizes YOLO instance segmentation labels.
* [rtsp_video.py](rtsp_video.py): RTSP video streaming.

## Installation and Usage

This repository uses `uv` for blazing-fast dependency management. The project decouples "computational backends" from "
tasks", allowing you to install exactly what you need based on your hardware.

1. **Install uv**
   ```bash
   pip install uv
   ```

2. **Sync Project Dependencies On-Demand**

   Choose **one** of the following commands based on your requirements:

    * **Basic image processing only (No ONNX inference)**
        ```bash
        uv sync
        ```
    * **CPU inference**
        ```bash
        uv sync --extra onnx-cpu
        ```
    * **GPU (NVIDIA) inference**
        ```bash
        uv sync --extra onnx-gpu
        ```
    * **inference SAM**
        ```bash
        uv sync --extra onnx-gpu --extra sam
        ```
    * **Install everything**
        ```bash
        uv sync --all-extras
        ```

3. **Run Scripts**

   Use `uv run` to execute scripts within the isolated environment you just synced:
   ```bash
   uv run <script_name>.py
   ```