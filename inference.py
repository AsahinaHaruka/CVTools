"""
@Project ：CVTools
@File ：tensorrt_inference.py
@Author ：Haruka
@Date ：2025/8/22 08:58
"""
import os

import onnxruntime as ort
import cv2
import numpy as np

from data_define import Area

# 使用字典映射动态获取模型输入的数据类型
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
            cache_dir="."
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


class AreaAvgInference(BashInference):
    def __init__(self, engine_path: str, areas: list[Area], confidence: float = 0.5,
                 class_num: int = 3):
        """
        Initialize the area average inference.
        :param engine_path: Path to the ONNX engine file.
        :param areas : Division of the area
        :param confidence: Confidence threshold for filtering detections.
        :param class_num: Number of classes for detection.
        """
        super().__init__(engine_path)
        self.confidence = confidence
        self.class_num = class_num
        self.areas = np.array([[area.start_x, area.start_y, area.end_x, area.end_y] for area in areas],
                              dtype=np.float32)

    def __call__(self, input_data: list[np.ndarray], raw=True) -> np.ndarray:
        """
        对输入图片进行推理，并进行NMS和置信度过滤。然后，筛选出那些落在感兴趣区域内的点，
        列表areas中的每一项表示一个感兴趣区域，定义见data_define Area，最后根据区域划分进行跨batch合并。
        在进行合并时，认为所有batch里anchors中心落在同一区域的是同一物体的边框,
        对同一物体的边框由加权平均确定，对同一物体的类别由出现的类别的大多数确定。
        :param input_data: Input data for inference. input_data :list[batch_size, height, width, channels]
        :return: Inference results with shape [ len(areas),5] where 5 = (x,y,x,y,class)， class=-1表示无检测
        """
        raw_output = super().__call__(input_data, raw=raw)  # [batch, 300, 6]

        result = process_detections(raw_output, self.areas, self.confidence, len(self.areas), self.class_num)

        return result


def process_detections(raw_output: np.ndarray, area_bounds: np.ndarray, confidence: float,
                       num_areas: int, class_num: int) -> np.ndarray:
    """
    处理已经过NMS的检测输出
    :param raw_output: [batch, 300, 6] where 6 = (x1,y1,x2,y2,score,class)
    :param area_bounds: [num_areas, 4] (start_x, start_y, end_x, end_y)
    :param confidence: 置信度阈值
    :param num_areas: 区域数量
    :param class_num: 类别数量
    :return: [num_areas, 5] (x1,y1,x2,y2,class) , class=-1表示无检测
    """
    valid_mask = raw_output[:, :, 4] >= confidence

    # 如果没有有效检测，直接返回空结果
    if not np.any(valid_mask):
        result = np.zeros((num_areas, 5), dtype=np.float32)
        result[:, 4] = -1.0  # 设置类别为-1表示没有检测到
        return result

    # 合并所有batch的检测结果
    # 检查是否有有效检测
    batch_indices, detection_indices = np.where(valid_mask)
    all_detections = raw_output[batch_indices, detection_indices]  # [n_valid, 6]

    # 从拼接后的数组中提取信息
    all_boxes = all_detections[:, :4]  # [n_valid, 4]
    all_scores = all_detections[:, 4]  # [n_valid]
    all_classes = all_detections[:, 5].astype(np.int32)  # [n_valid]

    # 计算中心点坐标 (从xyxy格式计算)
    all_centers = (all_boxes[:, :2] + all_boxes[:, 2:4]) / 2  # [n_valid, 2]

    # 区域分配
    # all_centers: [n_valid, 2], area_bounds: [num_areas, 4]
    centers_expanded = all_centers[:, np.newaxis, :]  # [n_valid, 1, 2]
    bounds_expanded = area_bounds[np.newaxis, :, :]  # [1, num_areas, 4]
    in_x_range = ((centers_expanded[:, :, 0] >= bounds_expanded[:, :, 0]) &
                  (centers_expanded[:, :, 0] <= bounds_expanded[:, :, 2]))
    in_y_range = ((centers_expanded[:, :, 1] >= bounds_expanded[:, :, 1]) &
                  (centers_expanded[:, :, 1] <= bounds_expanded[:, :, 3]))
    in_area_matrix = in_x_range & in_y_range  # [n_valid, num_areas]

    # 初始化结果
    result = np.zeros((num_areas, 5), dtype=np.float32)
    result[:, 4] = -1.0  # 默认类别为-1

    # 遍历每个区域进行处理
    for area_idx in range(num_areas):
        area_mask = in_area_matrix[:, area_idx]

        if not np.any(area_mask):
            continue

        # 获取该区域内的所有原始数据
        current_boxes = all_boxes[area_mask]
        current_scores = all_scores[area_mask]
        current_classes = all_classes[area_mask]

        # 类别投票 (确定主导类别)
        # 使用分数加权投票
        class_votes = np.bincount(current_classes, weights=current_scores, minlength=class_num)
        final_class = np.argmax(class_votes)

        # 只保留属于主导类别的框
        target_mask = (current_classes == final_class)

        target_boxes = current_boxes[target_mask]
        target_scores = current_scores[target_mask]

        # 加权平均

        weights = target_scores / np.sum(target_scores)
        weighted_box = np.sum(target_boxes * weights[:, np.newaxis], axis=0)

        result[area_idx, :4] = weighted_box
        result[area_idx, 4] = float(final_class)

    return result


class NumCountInference(BashInference):
    def __init__(self, engine_path: str, confidence: float = 0.5):
        """
        Initialize the num cunt inference.
        :param engine_path: Path to the ONNX engine file.
        :param confidence: Confidence threshold for filtering detections.
        """
        super().__init__(engine_path)
        self.confidence = confidence

    def __call__(self, input_data: list[np.ndarray], raw=False) -> int:
        """
        对输入图片进行推理，并进行NMS和置信度过滤(模型包含NMS)。然后，计算出钢坯数量，该数量值为batch中数量值的众数
        :param input_data: Input data for inference. input_data :list[batch_size, height, width, channels]
        :return: num count
        """

        # Get raw inference output: [batch, 300, 6] where 6 = (x1,y1,x2,y2,score,class)
        raw_output = super().__call__(input_data, raw=raw)

        # 所有batch的钢坯数量计数
        confidence_mask = raw_output[:, :, 4] >= self.confidence  # [batch, 300]
        batch_counts = np.sum(confidence_mask, axis=1)  # [batch]

        if len(batch_counts) == 1:
            return int(batch_counts[0])

        # 计算众数
        unique_counts, frequencies = np.unique(batch_counts, return_counts=True)
        mode_index = np.argmax(frequencies)

        return int(unique_counts[mode_index])
