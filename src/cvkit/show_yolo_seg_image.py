"""
@Project:  CVKit
@File:     show_yolo_seg_image.py
@Author:   Haruka
@Date:     2026/1/23 16:06
"""

import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

from cvkit.inference.yolo_inference import YoloSegInference
from cvkit.utils.logger import LoggerBuilder

logger = LoggerBuilder().get_logger(name="yolo_seg")


def draw_segmentation(
    image: np.ndarray, boxes: np.ndarray, masks: np.ndarray, conf_threshold: float = 0.5
) -> np.ndarray:
    """
    在图像上绘制分割和检测结果 (单张图片)。
    修改点：使用了预设的高对比度颜色列表，代替随机颜色。
    """
    vis_img = image.copy()

    # 如果没有检测结果，直接返回原图
    if boxes is None or len(boxes) == 0:
        return vis_img

    #  定义高对比度颜色列表 (BGR格式) ---

    palette = [
        (0, 165, 255),  # 4: Orange
        (255, 0, 255),  # 5: Purple
        (255, 255, 0),  # 6: Cyan
        (128, 0, 128),  # 7: Dark Purple
        (0, 128, 255),  # 8: Light Orange
        (203, 192, 255),  # 9: Pink
        (128, 128, 0),  # 10: Teal
        (0, 128, 0),  # 11: Dark Green
        (0, 0, 128),  # 12: Dark Red
        (238, 130, 238),  # 13: Violet
        (127, 255, 212),  # 14: Aquamarine
        (0, 0, 255),  # 0: Red
        (0, 255, 0),  # 1: Green
        (255, 0, 0),  # 2: Blue
        (0, 255, 255),  # 3: Yellow
    ]

    for i, box in enumerate(boxes):
        x1, y1, x2, y2, score, class_id = box

        # 过滤低置信度目标
        if score < conf_threshold:
            continue

        class_id = int(class_id)

        #  根据 class_id 从调色板取色 ---
        color = palette[class_id % len(palette)]

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        #  绘制掩码 (Mask)
        mask = masks[i]
        # 确保 mask 是 binary (0 或 1)，并转为 uint8
        mask = (mask > 0.5).astype(np.uint8)

        # 创建彩色遮罩
        colored_mask = np.zeros_like(vis_img, dtype=np.uint8)

        # 这里只给 mask 区域赋值颜色
        colored_mask[mask == 1] = color

        # 使用 addWeighted 进行半透明叠加
        mask_indices = mask == 1
        if np.any(mask_indices):
            vis_img[mask_indices] = cv2.addWeighted(
                vis_img[mask_indices], 0.5, colored_mask[mask_indices], 0.5, 0
            )

        # 绘制边界框 (Bounding Box)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

        # 绘制标签 (Label)
        label_text = f"Class {class_id}: {score:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # 标签背景色与框颜色一致
        cv2.rectangle(
            vis_img,
            (x1, y1 - label_height - baseline),
            (x1 + label_width, y1),
            color,
            -1,
        )

        text_color = (255, 255, 255)
        # 如果背景色太亮（例如青色或黄色），将文字改为黑色，方便阅读
        if sum(color) > 500:
            text_color = (0, 0, 0)

        cv2.putText(
            vis_img,
            label_text,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1,
        )

    return vis_img


def main(args: argparse.Namespace) -> None:
    """
    主函数，执行推理、绘制和保存。

    Args:
        args (argparse.Namespace): 命令行参数对象。
    """
    # 初始化推理引擎
    logger.info(f"Initializing Segmentation engine with model: {args.model}...")
    inference_engine = YoloSegInference(
        args.model, execution_provider=("CoreML", "CPUExecutionProvider")
    )

    # 准备输入输出目录
    os.makedirs(args.output, exist_ok=True)
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    # 检查输入是文件还是文件夹
    if os.path.isfile(args.input):
        image_files = [os.path.basename(args.input)]
        input_dir = os.path.dirname(args.input)
    else:
        image_files = [
            f for f in os.listdir(args.input) if f.lower().endswith(image_extensions)
        ]
        input_dir = args.input

    if not image_files:
        logger.error(f"No images found in {args.input}")
        return

    # 遍历图片进行处理
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Warning: Could not read image {image_file}. Skipping.")
            continue

        # 执行推理
        # res 包含 'box' 和 'masks'
        res = inference_engine([image], raw=False, return_boxes=True, return_masks=True)

        # 获取单张图片的结果 (batch index 0)
        boxes = res["box"][0]  # shape: (N, 6)
        masks = res["masks"][0]  # shape: (N, H, W)

        # 在图片上绘制结果
        annotated_image = draw_segmentation(
            image, boxes, masks, conf_threshold=args.conf
        )

        # 保存结果图片
        output_path = os.path.join(args.output, image_file)
        cv2.imwrite(output_path, annotated_image)

    logger.info(f"Inference complete. Annotated images are saved in '{args.output}'.")


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="YOLO Segmentation Inference Script.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the ONNX segmentation model file.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the image file or folder containing input images.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to the folder where results will be saved.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for visualization.",
    )

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    main_cli()
