"""
@Project ：CVTools
@File ：show_image.py
@Author ：Haruka
@Date ：2025/12/10 08:29
"""

import os
import argparse
import cv2
import numpy as np
from inference import BashInference
from tqdm import tqdm


def draw_detections(image, detections, conf_threshold=0.5):
    """
    在图像上绘制检测结果.
    :param image: 原始图像 (OpenCV BGR格式).
    :param detections: 单张图片的检测结果, shape [N, 6], 6 = (x1, y1, x2, y2, score, class_id).
    :param conf_threshold: 置信度阈值.
    :return: 绘制了检测框的图像.
    """
    # 为不同类别生成不同颜色
    np.random.seed(0)
    colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

    for det in detections:
        x1, y1, x2, y2, score, class_id = det
        if score < conf_threshold:
            continue

        box = np.array([x1, y1, x2, y2]).astype(int)
        class_id = int(class_id)
        color = colors[class_id % 100].tolist()

        # 绘制边界框
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)

        # 准备标签文本
        label = f"Class {class_id}: {score:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # 绘制标签背景
        cv2.rectangle(image, (box[0], box[1] - label_height - baseline), (box[0] + label_width, box[1]), color, -1)

        # 绘制标签文字
        cv2.putText(image, label, (box[0], box[1] - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image


def main(args):
    """主函数，执行推理、绘制和保存."""
    # 1. 初始化推理引擎
    print("Initializing inference engine...")
    inference_engine = BashInference(args.model)
    print("Engine initialized.")

    # 2. 准备输入输出目录
    os.makedirs(args.output, exist_ok=True)
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [f for f in os.listdir(args.input) if f.lower().endswith(image_extensions)]

    if not image_files:
        print(f"No images found in {args.input}")
        return

    # 3. 遍历图片进行处理
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(args.input, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_file}. Skipping.")
            continue

        # 执行推理
        detections = inference_engine([image])  # 输入为list

        # 在图片上绘制结果
        annotated_image = draw_detections(image, detections[0])  # 取batch中的第一个结果

        # 保存结果图片
        output_path = os.path.join(args.output, image_file)
        cv2.imwrite(output_path, annotated_image)

    print(f"\nInference complete. Annotated images are saved in '{args.output}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ONNX FP16 Inference Script for a folder of images.")
    parser.add_argument('--model', type=str, required=True, help="Path to the half-precision ONNX model file.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to the folder containing input images.")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Path to the folder where results will be saved.")

    args = parser.parse_args()
    main(args)
