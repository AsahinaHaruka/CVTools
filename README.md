# CVTools

这是一个用于常见计算机视觉任务的脚本合集。

## 文件说明

* [inference.py](inference.py): 用于ONNX模型推理的脚本(包含推理、区域平均的推理、计数推理)。
* [video2image.py](video2image.py): 将视频文件转换为图像序列,包含选点和透视变换
* [color_to_gray.py](color_to_gray.py): 将彩色图像转换为灰度图。
* [random_selection.py](random_selection.py): 从目录中随机选择一部分文件。
* [perspective_transformation.py](perspective_transformation.py): 对图像进行透视变换的类。

## 安装与使用

1. **安装依赖**

   本仓库使用 uv 来管理依赖，请确保您已安装 uv。

   ```bash
   pip install uv
   ```

2. **安装项目依赖**

   ```bash
   uv sync
   ```

3. **运行脚本**

   您可以使用 `uv run`来运行每个脚本。

   ```bash
   uv run  <script_name>.py
   ```

# CVTools

This is a collection of scripts for common computer vision tasks.

## File Descriptions

* [inference.py](inference.py): Script for ONNX model inference (includes inference, region average inference, and
  counting inference).
* [video2image.py](video2image.py): Converts a video file into an image sequence, including point selection and
  perspective transformation.
* [color_to_gray.py](color_to_gray.py): Converts a color image to grayscale.
* [random_selection.py](random_selection.py): Randomly selects a portion of files from a directory.
* [perspective_transformation.py](perspective_transformation.py): A class for performing perspective transformation on
  an image.

## Installation and Usage

1. **Install Dependencies**

   This repository uses `uv` to manage dependencies. Please ensure you have `uv` installed.

   ```bash
   pip install uv
   ```

2. **Install Project Dependencies**

   ```bash
   uv sync
   ```

3. **Run Scripts**

   You can use `uv run` to execute each script.

   ```bash
   uv run <script_name>.py
   ```