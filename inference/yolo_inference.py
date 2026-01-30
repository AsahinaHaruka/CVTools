"""
@Project ：CVTools
@File ：yolo_inference.py
@Author ：Haruka
@Date ：2025/8/22 08:58
"""
import numpy as np

from .base_inference import ONNXInference
from .image_processor import ImageProcessor
from utils.data_define import Area


class YoloObjInference:
    def __init__(self,
                 model_path: str,
                 enable_trt_profile: bool = False,
                 max_batch_size: int = 5,
                 opt_batch_size: int = 5,
                 input_image_size: tuple[int, int] | None = None,
                 target_long_side: int = 640,
                 execution_provider: tuple[str, ...] = ("trt", "cuda", "CoreML", "cpu")):
        """初始化 YOLO 推理会话。

        加载 YOLO ONNX 模型并配置推理会话选项。支持 TensorRT 执行提供程序及其动态形状配置。
        如果启用了 TensorRT 配置文件 (`enable_trt_profile=True`)，将根据提供的批处理大小和图像尺寸参数
        构建优化配置文件。

        Args:
            model_path (str): ONNX 模型文件的路径，该模型应包含 NMS (Non-Maximum Suppression) 操作。
            enable_trt_profile (bool, optional): 是否启用 TensorRT 动态形状配置文件生成。
                如果为 True，将根据 batch size 和尺寸参数预热 TensorRT 引擎缓存。
                当此参数为 True 时，无论是否提供 `input_image_size`，`fix_image` 都将被设置为 True。
                默认为 False。
            max_batch_size (int, optional): 预期的最大 Batch Size。
                仅在 `enable_trt_profile=True` 且模型输入包含动态 Batch 维度时生效。默认为 5。
            opt_batch_size (int, optional): 预期的最优 Batch Size。
                仅在 `enable_trt_profile=True` 且模型输入包含动态 Batch 维度时生效。默认为 5。
            input_image_size (tuple[int, int] | None, optional): 原始输入图像的分辨率 (Width, Height)。
                如果提供，将基于 `target_long_side` 计算最佳矩形推理尺寸 (Rectangular Inference)，此时 fix_image 为 True。
                如果不提供，默认使用 `target_long_side` 作为边长的正方形尺寸，此时 fix_image 为 False。
                默认为 None。
            target_long_side (int, optional): 预期的输入图像长边尺寸。
                用于计算实际推理时的输入分辨率。默认为 640。
            execution_provider (tuple[str, ...], optional): 需要扫描的后端执行提供者列表。
                默认为 ("trt", "cuda", "CoreML", "cpu")。

        Raises:
            FileNotFoundError: 如果 `model_path` 指定的文件不存在。
        """
        self.model = ONNXInference(model_path=model_path,
                                   stride=32,
                                   enable_trt_profile=enable_trt_profile,
                                   max_batch_size=max_batch_size,
                                   opt_batch_size=opt_batch_size,
                                   input_image_size=input_image_size,
                                   target_long_side=target_long_side,
                                   execution_provider=execution_provider
                                   )

        self.image_processor = ImageProcessor(target_size=self.model.img_size,
                                              stride=self.model.stride,
                                              is_fixed_size=self.model.fix_image,
                                              fill_value=144,
                                              dtype=self.model.input_meta[0]['type'])

    def __call__(self, input_data: list[np.ndarray] | np.ndarray, raw: bool = False) -> np.ndarray:
        """执行 YOLO 模型的推理。

        该方法首先对输入图像数据进行预处理，然后将处理后的图像输入模型进行推理。
        根据 `raw` 参数，选择是否将推理结果（边界框）还原到原始图像坐标系。

        Args:
            input_data (list[np.ndarray] | np.ndarray): 输入图像数据。
                可以是单个 NumPy 数组 `(H, W)` 或 `(H, W, C)`，
                也可以是包含多个 NumPy 数组的列表，每个数组代表一张图像。
            raw (bool, optional): 是否返回原始模型输出。
                如果为 True，则返回模型原始输出的边界框（未经过后处理）；
                如果为 False，边界框将被还原到原始图像坐标系。默认为 False。

        Returns:
            np.ndarray: 推理结果。形状为 `(batch, max_detections, output_dim)`，
                        其中 `output_dim` 通常为 6，表示 `(x1, y1, x2, y2, score, class_id)`。
        """
        # 预处理输入数据
        processed_input, transform_params = self.image_processor(input_data)

        # 执行推理
        outputs = self.model(processed_input)[0].astype(np.float32)
        return outputs if raw else self.image_processor.restore_boxes(outputs, transform_params)


class AreaAvgInference(YoloObjInference):
    def __init__(self,
                 model_path: str,
                 areas: list[Area],
                 confidence: float = 0.5,
                 class_num: int = 3,
                 enable_trt_profile: bool = True,
                 max_batch_size: int = 5,
                 opt_batch_size: int = 5,
                 input_image_size: tuple[int, int] | None = None,
                 target_long_side: int = 640):
        """初始化区域平均推理会话。

        Args:
            model_path (str): ONNX 模型文件的路径。
            areas (list[Area]): 预定义区域的列表。每个 Area 对象应包含 start_x, start_y, end_x, end_y 属性。
            confidence (float, optional): 置信度阈值。低于此值的检测结果将被忽略。默认为 0.5。
            class_num (int, optional): 模型输出的类别总数。用于类别投票时的 `minlength` 参数。默认为 3。
            enable_trt_profile (bool, optional): 是否启用 TensorRT 动态形状配置文件生成。
                如果为 True，将根据 batch size 和尺寸参数预热 TensorRT 引擎缓存。
                当此参数为 True 时，无论是否提供 `input_image_size`，`fix_image` 都将被设置为 True。
                默认为 True。
            max_batch_size (int, optional): 预期的最大 Batch Size。默认为 5。
            opt_batch_size (int, optional): 预期的最优 Batch Size。默认为 5。
            input_image_size (tuple[int, int] | None, optional): 原始输入图像的分辨率 (Width, Height)。
                如果提供，将基于 `target_long_side` 计算最佳矩形推理尺寸 (Rectangular Inference)，此时 fix_image 为 True。
                如果不提供，默认使用 `target_long_side` 作为边长的正方形尺寸，此时 fix_image 为 False。
                默认为 None。
            target_long_side (int, optional): 预期的输入图像长边尺寸。
                用于计算实际推理时的输入分辨率。默认为 640。

        Raises:
            FileNotFoundError: 如果 `model_path` 指定的文件不存在。
        """
        super().__init__(model_path=model_path,
                         enable_trt_profile=enable_trt_profile,
                         max_batch_size=max_batch_size,
                         opt_batch_size=opt_batch_size,
                         input_image_size=input_image_size,
                         target_long_side=target_long_side
                         )
        self.confidence = confidence
        self.class_num = class_num
        self.areas = np.array([[area.start_x, area.start_y, area.end_x, area.end_y] for area in areas],
                              dtype=np.float32)

    def __call__(self, input_data: list[np.ndarray] | np.ndarray, raw: bool = False) -> np.ndarray:
        """对输入图片进行推理，然后根据预定义区域进行结果合并。

        该方法首先调用父类的 `__call__` 方法获取原始的检测结果。
        接着，它会筛选出置信度高于阈值的检测框，并根据这些检测框的中心点判断它们属于哪个预定义区域。
        对于每个区域，它会合并所有批次中落入该区域的检测结果，通过加权平均确定最终的边界框，
        并通过加权投票确定主导类别。

        Args:
            input_data (list[np.ndarray] | np.ndarray): 输入图像数据。
                可以是单个 NumPy 数组 `(H, W)` 或 `(H, W, C)`，
                也可以是包含多个 NumPy 数组的列表。
            raw (bool, optional): 是否返回原始模型输出。
                如果为 True，则返回模型原始输出的边界框（未经过后处理）；
                否则，边界框将被还原到原始图像坐标系。默认为 False。
                                  注意：此处的 `raw` 参数传递给 `super().__call__`，
                                  但 `process_detections` 始终处理还原后的坐标。

        Returns:
            np.ndarray: 经过区域合并和处理后的检测结果，形状为 `(len(self.areas), 5)`。
                        其中 5 代表 `(x1, y1, x2, y2, class_id)`。
                        如果某个区域没有检测到目标，其类别将为 -1。
        Note:
            此函数假设 NMS 内置于模型之中。
        """
        raw_output = super().__call__(input_data, raw=raw)  # [batch, 300, 6]

        result = process_detections(raw_output, self.areas, self.confidence, len(self.areas), self.class_num)

        return result


def process_detections(raw_output: np.ndarray, area_bounds: np.ndarray, confidence: float,
                       num_areas: int, class_num: int) -> np.ndarray:
    """处理经过NMS的检测输出，并根据预定义区域进行加权平均和类别投票。

    该函数将所有批次中置信度高于阈值的检测结果合并，然后根据每个检测框的中心点，
    将其分配到对应的预定义区域。对于每个区域，它会计算落入该区域的所有检测框的
    加权平均边界框，并通过加权投票确定该区域的主导类别。

    Args:
        raw_output (np.ndarray): 模型的原始检测输出，形状为 `(batch_size, max_detections, 6)`。
            其中 6 代表 `(x1, y1, x2, y2, score, class_id)`。
        area_bounds (np.ndarray): 预定义区域的边界坐标，形状为 `(num_areas, 4)`。
            其中 4 代表 `(start_x, start_y, end_x, end_y)`。
        confidence (float): 检测结果的置信度阈值。低于此阈值的检测将被忽略。
        num_areas (int): 预定义区域的数量。
        class_num (int): 模型输出的类别总数。用于类别投票时的 `minlength` 参数。

    Returns:
        np.ndarray: 经过处理后的检测结果，形状为 `(num_areas, 5)`。
            其中 5 代表 `(x1, y1, x2, y2, class_id)`。
                    如果某个区域没有检测到目标，其 `class_id` 将为 -1。

    Note:
        此函数假设 `raw_output` 中的边界框坐标已经还原到原始图像尺寸，
        或者至少是统一的坐标系，以便与 `area_bounds` 进行比较。
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


class NumCountInference(YoloObjInference):
    def __init__(self,
                 model_path: str,
                 confidence: float = 0.5,
                 enable_trt_profile: bool = True,
                 max_batch_size: int = 5,
                 opt_batch_size: int = 5,
                 input_image_size: tuple[int, int] | None = None,
                 target_long_side: int = 640
                 ):
        """初始化数量统计推理会话。

        Args:
            model_path (str): ONNX 模型文件的路径。
            confidence (float, optional): 置信度阈值。默认为 0.5。
            enable_trt_profile (bool, optional): 是否启用 TensorRT 动态形状配置文件生成。
                如果为 True，将根据 batch size 和尺寸参数预热 TensorRT 引擎缓存。
                当此参数为 True 时，无论是否提供 `input_image_size`，`fix_image` 都将被设置为 True。
                默认为 True。
            max_batch_size (int, optional): 预期的最大 Batch Size。默认为 5。
            opt_batch_size (int, optional): 预期的最优 Batch Size。默认为 5。
            input_image_size (tuple[int, int] | None, optional): 原始输入图像的分辨率 (Width, Height)。
                如果提供，将基于 `target_long_side` 计算最佳矩形推理尺寸 (Rectangular Inference)，此时 fix_image 为 True。
                如果不提供，默认使用 `target_long_side` 作为边长的正方形尺寸，此时 fix_image 为 False。
                默认为 None。
            target_long_side (int, optional): 预期的输入图像长边尺寸。
                用于计算实际推理时的输入分辨率。默认为 640。

        Raises:
            FileNotFoundError: 如果 `model_path` 指定的文件不存在。
        """

        super().__init__(model_path=model_path,
                         enable_trt_profile=enable_trt_profile,
                         max_batch_size=max_batch_size,
                         opt_batch_size=opt_batch_size,
                         input_image_size=input_image_size,
                         target_long_side=target_long_side
                         )
        self.confidence = confidence

    def __call__(self, input_data: list[np.ndarray] | np.ndarray, raw: bool = True) -> int:
        """对输入图片进行推理，并进行NMS和置信度过滤，然后计算目标数量。

        该方法首先对输入数据进行预处理，然后执行模型推理。
        推理结果经过置信度过滤后，统计每个批次中检测到的目标数量。
        如果批次大小为1，则直接返回该批次的目标数量。
        如果批次大小大于1，则计算所有批次目标数量的众数作为最终结果。

        Args:
            input_data (list[np.ndarray] | np.ndarray): 输入图像数据。
                可以是单个 NumPy 数组 `(H, W)` 或 `(H, W, C)`，
                也可以是包含多个 NumPy 数组的列表。
            raw (bool, optional): 是否返回原始模型输出。默认为 True。

        Returns:
            int: 检测到的目标数量的众数。
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


class YoloSegInference:
    def __init__(self,
                 model_path: str,
                 enable_trt_profile: bool = False,
                 max_batch_size: int = 5,
                 opt_batch_size: int = 5,
                 input_image_size: tuple[int, int] | None = None,
                 target_long_side: int = 640,
                 execution_provider: tuple[str, ...] = ("trt", "cuda", "CoreML", "cpu"),
                 uniform_transform: bool = False):
        """初始化 YOLO 实例分割推理会话。

        加载 YOLO ONNX 模型并配置推理会话选项。支持 TensorRT 执行提供程序及其动态形状配置。
        如果启用了 TensorRT 配置文件 (`enable_trt_profile=True`)，将根据提供的批处理大小和图像尺寸参数
        构建优化配置文件。

        Args:
            model_path (str): ONNX 模型文件的路径，该模型应包含 NMS 操作。
            enable_trt_profile (bool, optional): 是否启用 TensorRT 动态形状配置文件生成。
                如果为 True，将根据 batch size 和尺寸参数预热 TensorRT 引擎缓存。
                当此参数为 True 时，无论是否提供 `input_image_size`，`fix_image` 都将被设置为 True。
                默认为 False。
            max_batch_size (int, optional): 预期的最大 Batch Size。默认为 5。
            opt_batch_size (int, optional): 预期的最优 Batch Size。默认为 5。
            input_image_size (tuple[int, int] | None, optional): 原始输入图像的分辨率 (Width, Height)。
                如果提供，将基于 `target_long_side` 计算最佳矩形推理尺寸 (Rectangular Inference)，此时 fix_image 为 True。
                如果不提供，默认使用 `target_long_side` 作为边长的正方形尺寸，此时 fix_image 为 False。
                默认为 None。
            target_long_side (int, optional): 预期的输入图像长边尺寸。
                用于计算实际推理时的输入分辨率。默认为 640。
            execution_provider (tuple[str, ...], optional): 需要扫描的后端执行提供者列表。
                默认为 ("trt", "cuda", "CoreML", "cpu")。
            uniform_transform (bool, optional): 是否启用统一变换优化。
                如果为 True，假设 Batch 内所有图片可以使用完全相同的预处理和后处理（如视频流），
        Raises:
            FileNotFoundError: 如果 `model_path` 指定的文件不存在。
        """
        self.model = ONNXInference(model_path=model_path,
                                   stride=32,
                                   enable_trt_profile=enable_trt_profile,
                                   max_batch_size=max_batch_size,
                                   opt_batch_size=opt_batch_size,
                                   input_image_size=input_image_size,
                                   target_long_side=target_long_side,
                                   execution_provider=execution_provider
                                   )

        self.image_processor = ImageProcessor(target_size=self.model.img_size,
                                              stride=self.model.stride,
                                              is_fixed_size=self.model.fix_image,
                                              fill_value=144,
                                              dtype=self.model.input_meta[0]['type'],
                                              uniform_transform=uniform_transform)

    def __call__(self, input_data: list[np.ndarray] | np.ndarray,
                 conf_threshold: float = 0.25,
                 mask_threshold: float = 0.5,
                 return_boxes: bool = True,
                 return_masks: bool = True,
                 raw: bool = False) -> dict[str, list[np.ndarray] | np.ndarray]:
        """执行 YOLO 分割模型的推理。

        该方法首先对输入图像数据进行预处理，然后将处理后的图像输入模型进行推理。
        根据 `raw` 参数，选择是否将推理结果（边界框和mask）还原到原始图像坐标系。

        Args:
            input_data (list[np.ndarray] | np.ndarray): 输入图像数据。
                可以是单个 NumPy 数组 `(H, W)` 或 `(H, W, C)`，
                也可以是包含多个 NumPy 数组的列表，每个数组代表一张图像。
            conf_threshold (float): 置信度阈值。默认为 0.25。
            mask_threshold (float, optional): Mask 二值化的阈值。默认为 0.5。
            return_boxes (bool, optional): 是否在结果中包含边界框。默认为 True。
            return_masks (bool, optional): 是否在结果中包含分割掩码。默认为 True。
            raw (bool, optional): 是否返回原始模型输出。
                如果为 True，则返回模型原始输出的边界框和掩码（未经过后处理）；
                否则，边界框和掩码将被还原到原始图像坐标系。默认为 False。

        Returns:
            dict[str, list[np.ndarray] | np.ndarray]: 包含推理结果的字典。
                - 如果 `raw=True`，返回包含原始 `'boxes'`, `'mask_coefficients'`, `'protos'` 的字典。
                - 否则返回包含 `'box'` 和 `'masks'` 的字典：
                    - `'box'`: `list[np.ndarray]`，每个元素为 `(N, 6)`，格式为 `[x1, y1, x2, y2, score, class]`。
                    - `'masks'`: `list[np.ndarray]`，每个元素为 `(N, H, W)` 的二值掩码。
        """
        # 预处理输入数据
        processed_input, transform_params = self.image_processor(input_data)

        # 执行推理
        outputs =self.model(processed_input)

        detections, protos = outputs[0], outputs[1]

        if raw:
            return {
                'boxes': detections[..., :6],
                'mask_coefficients': detections[..., 6:],
                'protos': protos
            }

        batch_raw_boxes = []  # 收集原始检测框
        batch_raw_masks = []  # 收集原始 Mask

        batch_size = detections.shape[0]

        for i in range(batch_size):
            det = detections[i]  # [N, 6+]
            proto = protos[i]  # [32, MH, MW]

            # 阈值过滤
            keep = det[:, 4] >= conf_threshold
            det = det[keep]

            if len(det) == 0:
                if return_boxes or return_masks:
                    batch_raw_boxes.append(np.zeros((0, det.shape[1]), dtype=np.float32))

                if return_masks:
                    _, mh, mw = proto.shape
                    batch_raw_masks.append(np.zeros((0, mh, mw), dtype=np.float32))
                continue

            if return_boxes or return_masks:
                batch_raw_boxes.append(det)

            if return_masks:
                mask_coefficients = det[:, 6:]
                _, mh, mw = proto.shape

                # Matmul: [M, 32] @ [32, H*W] -> [M, H*W]
                masks_flat = np.matmul(mask_coefficients, proto.reshape(32, -1))

                # Sigmoid
                clip_limit = 9 if masks_flat.dtype == np.float16 else 80
                masks_flat = masks_flat.clip(-clip_limit, clip_limit)
                masks_flat = 1 / (1 + np.exp(-masks_flat))

                # Reshape -> [M, H, W] & Append
                batch_raw_masks.append(masks_flat.reshape(-1, mh, mw))

        res = {}

        if return_boxes:
            restored_boxes_list = self.image_processor.restore_boxes(
                batch_raw_boxes,
                transform_params
            )

            # 只需要前6列 (xyxy, conf, cls)
            res['box'] = [b[:, :6] for b in restored_boxes_list]

        if return_masks:
            res['masks'] = self.image_processor.restore_masks(
                masks=batch_raw_masks,  # List[np.ndarray]
                transform_params=transform_params,
                boxes=[b[:, :4] for b in batch_raw_boxes],
                mask_threshold=mask_threshold,
                uniform_transform=False
            )

        return res
