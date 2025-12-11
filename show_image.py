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
import onnxruntime as ort
from tqdm import tqdm

DTYPE_MAPPING = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(int8)": np.int8
}


class BashInference:
    def __init__(self, model_path: str):
        """
        Initialize the ONNX inference session.
        :param model_path: Path to the ONNX model file.
        """
        try:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            cache_dir = f"./trt_cache/{model_name}"
            os.makedirs(cache_dir, exist_ok=True)
        except:
            cache_dir = "."
        providers = [
            ('TensorrtExecutionProvider', {
                'device_id': 0,
                'trt_max_workspace_size': 4294967296,  # TensorRT最大内存 (4GB)
                'trt_fp16_enable': True,  # 启用 FP16 加速
                'trt_engine_cache_enable': True,  # 启用 TensorRT 引擎缓存
                'trt_engine_cache_path': cache_dir,  # 缓存路径
            }),
            'CUDAExecutionProvider',
            'CPUExecutionProvider']
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.log_severity_level = 3  # 只输出警告和错误

        # 创建ONNX Runtime会话
        self.session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
        print(f"ONNX run on {self.session.get_providers()[0]}")

        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # 从模型获取输入尺寸信息
        input_shape = self.session.get_inputs()[0].shape
        if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
            self.img_size = [input_shape[2], input_shape[3]]
        else:
            self.img_size = [640, 640]  # 默认尺寸

        self.stride = 32  # 默认stride
        # 动态获取模型输入的数据类型
        input_type = self.session.get_inputs()[0].type
        self.dtype = DTYPE_MAPPING.get(input_type, np.float32)

    def __call__(self, input_data: list[np.ndarray], raw=True) -> np.ndarray:
        """
        Perform inference on the input data.
        :param input_data: Input data for inference. input_data :list[batch_size, height, width, channels]
        :param raw: If True, return results in original image coordinates; otherwise, return in model output coordinates.
        :return: Inference results with shape [batch, self.max_detections,  self.output_dim],default self.output_dim=6, where 6 = (x,y,x,y,score,class)
        """
        # 预处理输入数据
        processed_input, transform_params = self._preprocess(input_data)

        # 执行推理
        outputs = self.session.run([self.output_name], {self.input_name: processed_input})
        return self._convert_to_original_coords(outputs[0].astype(np.float32), transform_params) \
            if raw else outputs[0].astype(np.float32)

    def trans_img(self, img: np.ndarray) -> tuple[np.ndarray, dict]:
        """Resize and pad an image for object detection"""
        shape = img.shape[:2]
        r = min(self.img_size[0] / shape[0], self.img_size[1] / shape[1])  # 缩放比例
        new_pad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 缩放后的宽高
        dw, dh = self.img_size[1] - new_pad[0], self.img_size[0] - new_pad[1]  # 填充量
        dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_pad:  # resize
            transformed_img = cv2.resize(img, new_pad, interpolation=cv2.INTER_LINEAR)
            if transformed_img.ndim == 2:
                transformed_img = transformed_img[..., None]
        else:
            transformed_img = img.copy()
            if transformed_img.ndim == 2:
                transformed_img = transformed_img[..., None]

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        h, w, c = transformed_img.shape
        if c == 3:
            transformed_img = cv2.copyMakeBorder(
                transformed_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,) * 3
            )
        else:  # multispectral
            pad_img = np.full((h + top + bottom, w + left + right, c), fill_value=114, dtype=transformed_img.dtype)
            pad_img[top: top + h, left: left + w] = transformed_img
            transformed_img = pad_img
        transform_params = {
            'orig_shape': shape,
            'scale': r,
            'padding': (left, top)
        }
        return transformed_img, transform_params

    def _preprocess(self, input_data: list[np.ndarray]) -> tuple[np.ndarray, list[dict]]:
        """
        预处理输入数据
        :param input_data: 输入图像列表
        :return: 预处理后的numpy数组
        """
        transformed_data = []
        transform_params = []

        for img in input_data:
            t_img, t_params = self.trans_img(img)
            transformed_data.append(t_img)
            transform_params.append(t_params)

        input_tensor = np.stack(transformed_data)
        if input_tensor.shape[-1] == 3:
            input_tensor = input_tensor[..., ::-1]  # BGR to RGB
        input_tensor = input_tensor.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
        input_tensor = np.ascontiguousarray(input_tensor)

        input_tensor = input_tensor.astype(np.float32) / 255.0
        if self.dtype != np.float32:
            input_tensor = input_tensor.astype(self.dtype)

        return input_tensor, transform_params

    @staticmethod
    def _convert_to_original_coords(detections: np.ndarray, transform_params: list) -> np.ndarray:
        """
        将检测结果转换回原始图像坐标系
        :param detections: 模型输出的检测结果 [batch, max_detections, output_dim]
        :param transform_params: 每个图像的变换参数
        :return: 原始坐标系下的检测结果
        """
        batch_size = detections.shape[0]
        result = detections.copy()

        for i in range(batch_size):
            valid_mask = detections[i, :, 4] > 0  # 假设第5列是置信度分数

            if not np.any(valid_mask):
                continue  # 如果没有有效检测，跳过当前batch

            # 获取当前图像的变换参数
            params = transform_params[i]
            r = params['scale']  # 缩放比例
            pad_w, pad_h = params['padding']  # 填充量

            # 提取检测框坐标并转换
            valid_boxes = result[i, valid_mask, :4]  # (x, y, x, y)

            # 坐标转换：减去填充再除以缩放比例
            valid_boxes[:, 0] = (valid_boxes[:, 0] - pad_w) / r  # x1
            valid_boxes[:, 1] = (valid_boxes[:, 1] - pad_h) / r  # y1
            valid_boxes[:, 2] = (valid_boxes[:, 2] - pad_w) / r  # x2
            valid_boxes[:, 3] = (valid_boxes[:, 3] - pad_h) / r  # y2

            # 更新结果
            result[i, valid_mask, :4] = valid_boxes

        return result

    def __del__(self):
        """清理资源"""
        try:

            if hasattr(self, 'session'):
                del self.session

        except Exception:
            pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()
        return False


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
    parser.add_argument('--input', type=str, required=True, help="Path to the folder containing input images.")
    parser.add_argument('--output', type=str, required=True, help="Path to the folder where results will be saved.")

    args = parser.parse_args()
    main(args)
