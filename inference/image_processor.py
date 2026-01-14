"""
@Project : CVTools 
@File : image_processor.py
@Author : Haruka
@Date : 2026/1/9 13:41 
"""
import cv2
import numpy as np


def zero2one(input_tensor: np.ndarray) -> np.ndarray:
    """
    将输入张量归一化到 [0, 1] 范围。

    Args:
        input_tensor (np.ndarray): 待归一化的 NumPy 数组，通常为图像数据。

    Returns:
        np.ndarray: 归一化后的 NumPy 数组，数据类型为 float32。
    """
    return input_tensor.astype(np.float32) / 255.0


def minus_one2one(input_tensor: np.ndarray) -> np.ndarray:
    """
    将输入张量归一化到 [-1, 1] 范围。

    Args:
        input_tensor (np.ndarray): 待归一化的 NumPy 数组，通常为图像数据。

    Returns:
        np.ndarray: 归一化后的 NumPy 数组，数据类型为 float32。
    """
    input_tensor = input_tensor.astype(np.float32) / 255.0
    return (input_tensor - 0.5) / 0.5


def imagenet(input_tensor: np.ndarray) -> np.ndarray:
    """
    使用 ImageNet 的均值和标准差对输入张量进行归一化。

    Args:
        input_tensor (np.ndarray): 待归一化的 NumPy 数组，通常为图像数据，
            期望形状为 (N, C, H, W) 或 (C, H, W)，且像素值在 [0, 255] 范围内。

    Returns:
        np.ndarray: 归一化后的 NumPy 数组，数据类型为 float32。
    """
    input_tensor = input_tensor.astype(np.float32) / 255.0

    # 确保 mean 和 std 的形状与输入张量兼容，以便进行广播
    # 如果输入是 (N, C, H, W)，则 mean/std 形状为 (1, C, 1, 1)
    # 如果输入是 (C, H, W)，则 mean/std 形状为 (C, 1, 1)
    num_channels = input_tensor.shape[1] if input_tensor.ndim == 4 else input_tensor.shape[0]
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # 调整形状以匹配输入张量的维度，通常是 (1, C, 1, 1) 或 (C, 1, 1)
    mean = mean.reshape(1, num_channels, 1, 1) if input_tensor.ndim == 4 else mean.reshape(num_channels, 1, 1)
    std = std.reshape(1, num_channels, 1, 1) if input_tensor.ndim == 4 else std.reshape(num_channels, 1, 1)
    return (input_tensor - mean) / std

NORM_CALL={
    '0_1': zero2one,
    '-1_1': minus_one2one,
    'imagenet':imagenet
}


class ImageProcessor:
    def __init__(self,
                 target_size: int | tuple[int, int] = 640,
                 stride: int = 32,
                 is_fixed_size: bool = False,
                 norm_type: str = "0_1",
                 fill_value: int = 114,
                 dtype: np.dtype = np.float32):
        """
        图像预处理与后处理工具类。

        Args:
            target_size (int | tuple[int, int]): 目标尺寸。
                如果为 int，表示目标图像的目标长边。
                如果为 tuple[int, int]，表示目标图像的 (height, width)。
                是否填充短边取决于 `is_fixed_size`
                默认为 640
            stride (int): 模型步长，用于计算 padding 对齐。默认为 32。
            is_fixed_size (bool): 是否强制缩放到固定尺寸。
                如果为 True，图像将被缩放并填充到 `target_size` 指定的绝对尺寸。
                如果为 False，图像长边缩放到 `target_size`，短边自然缩放，
                然后填充到 `stride` 的倍数。默认为 False。
            norm_type (str): 归一化类型，可选 '0_1', '-1_1', 'imagenet'。默认为 "0_1"。
            fill_value (int): Letterbox 填充时的像素值。默认为 114。
            dtype (np.dtype): 最终输出 NumPy 数组的数据类型。默认为 np.float32。
        """
        self.stride = stride
        self.is_fixed_size = is_fixed_size
        self.fill_value = fill_value
        self.dtype = dtype

        self._norm_function=NORM_CALL.get(norm_type)

        # 统一处理 target_size
        if isinstance(target_size, int):
            self.target_h, self.target_w = target_size, target_size
            self.long_side = target_size
        else:
            self.target_h, self.target_w = target_size
            self.long_side = max(self.target_h, self.target_w)

    def letterbox(self, img: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        对单张图片进行 Letterbox 处理，包括缩放和填充。

        根据 `is_fixed_size` 的设置，图片会被缩放到目标尺寸并填充，
        或者长边缩放到 `target_long_side`，短边自然缩放后填充到 `stride` 的倍数。

        Args:
            img (np.ndarray): 输入图像，形状为 (H, W) 或 (H, W, C)。

        Returns:
            tuple[np.ndarray, dict]: 包含两个元素的元组。
                - np.ndarray: 经过 Letterbox 处理后的图像，形状为 (H', W', C')。
                - dict: 包含变换参数的字典，键包括 'orig_shape' (原始图像形状), 'scale' (缩放比例), 'padding' (左上角填充量)。
        """
        """单张图片的 Letterbox 处理 """

        shape = img.shape[:2]

        if self.is_fixed_size:
            # 强制适配到绝对尺寸 (Target H, Target W)
            r = min(self.target_h / shape[0], self.target_w / shape[1])  # 缩放比例
            new_pad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 缩放后的宽高
            dw, dh = self.target_w - new_pad[0], self.target_h - new_pad[1]  # 填充量

        else:
            # 标准动态矩形,只把长边缩放到 long_side，短边自然缩放
            r = self.long_side / max(shape[0], shape[1])

            new_pad = int(round(shape[1] * r)), int(round(shape[0] * r))

            # 计算动态 padding: 只需要补齐到 stride 的倍数
            dw = self.long_side - new_pad[0]
            dh = self.long_side - new_pad[1]
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # 取模

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_pad:  # resize
            transformed_img = cv2.resize(img, new_pad, interpolation=cv2.INTER_LINEAR)

        else:
            transformed_img = img.copy()

        # 处理灰度图 (H, W) -> (H, W, 1)
        if transformed_img.ndim == 2:
            transformed_img = transformed_img[..., None]

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        h, w, c = transformed_img.shape

        # 针对不同通道数的通用处理
        if c == 3:
            transformed_img = cv2.copyMakeBorder(
                transformed_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(self.fill_value,) * 3
            )
        else:  # multispectral
            pad_img = np.full((h + top + bottom, w + left + right, c), fill_value=self.fill_value,
                              dtype=transformed_img.dtype)
            pad_img[top: top + h, left: left + w] = transformed_img
            transformed_img = pad_img

        transform_params = {
            'orig_shape': shape,
            'scale': r,
            'padding': (left, top)
        }
        return transformed_img, transform_params

    def __call__(self, input_data: list[np.ndarray] | np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        对输入图像数据进行预处理，包括 Letterbox 缩放、通道转换、维度重排和归一化。

        Args:
            input_data (list[np.ndarray] | np.ndarray): 输入图像数据。
                可以是单个 NumPy 数组 (H, W) 或 (H, W, C)，
                也可以是包含多个 NumPy 数组的列表。

        Returns:
            tuple[np.ndarray, list[dict]]: 包含两个元素的元组。
                - np.ndarray: 经过预处理的图像张量，形状为 (N, C, H, W)，数据类型为 self.dtype。
                - list[dict]: 包含每张图像原始形状、缩放比例和填充量的字典列表，
                              用于后续的后处理（如坐标还原）。
        """
        """
        """
        if isinstance(input_data, np.ndarray):
            # 如果是np.ndarray且为 (H, W) 或 (H, W, C)，说明是单张图 -> 包装成 list
            if input_data.ndim == 2 or input_data.ndim == 3:
                input_data = [input_data]

        transformed_data = []
        transform_params = []

        for img in input_data:
            t_img, t_params = self.letterbox(img)
            transformed_data.append(t_img)
            transform_params.append(t_params)

        input_tensor = np.stack(transformed_data)
        if input_tensor.shape[-1] == 3:
            input_tensor = input_tensor[..., ::-1]  # BGR to RGB
        input_tensor = input_tensor.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
        input_tensor = np.ascontiguousarray(input_tensor)

        input_tensor=self._norm_function(input_tensor)

        return input_tensor.astype(self.dtype), transform_params

    @staticmethod
    def convert_to_original_coords(detections: np.ndarray, transform_params: list) -> np.ndarray:
        """
        将检测结果从模型输出坐标系转换回原始图像坐标系（YOLO系列）。

        此方法用于在模型推理后，将检测到的边界框坐标从经过 Letterbox 处理后的图像尺寸
        还原到原始输入图像的尺寸。

        Args:
            detections (np.ndarray): 模型输出的检测结果，形状为 `[batch, max_detections, output_dim]`。
                                     其中 `output_dim` 至少包含 `(x1, y1, x2, y2, score, ...)`。
                                     假设第5列（索引为4）是置信度分数，用于过滤无效检测。
            transform_params (list[dict]): 包含每个图像预处理时使用的变换参数的字典列表。
                                           每个字典应包含 'scale' (缩放比例) 和 'padding' (左上角填充量)。
        Returns:
            np.ndarray: 转换回原始图像坐标系后的检测结果，形状与 `detections` 相同。
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

    @staticmethod
    def convert_normalized_boxes(boxes: np.ndarray,
                                 transform_params: list,
                                 input_shape: tuple[int, int],
                                 box_format: str = 'xyxy') -> np.ndarray:
        """
        将归一化后的边界框坐标转换回原始图像坐标系。

        此方法处理以下转换步骤：
        1. **反归一化**: 如果边界框坐标是 [0, 1] 范围内的归一化值，则将其乘以模型的输入尺寸 (model_w, model_h)。
           通过检查坐标的最大值（如果小于等于 1.5，则认为是归一化数据）来自动判断。
        2. **格式转换**: 如果 `box_format` 为 'cxcywh' (中心点坐标和宽高)，则将其转换为 'xyxy' (左上角和右下角坐标)。
        3. **还原到原图**: 减去 Letterbox 预处理时添加的填充 (padding)，然后除以缩放比例 (scale)，
           将坐标从 Letterbox 图像尺寸还原到原始图像尺寸。

        Args:
            boxes (np.ndarray): 模型的输出边界框，形状为 `(batch_size, num_boxes, 4)`。
                                坐标可以是归一化的 (0-1) 或非归一化的，格式可以是 'xyxy' 或 'cxcywh'。
            transform_params (list[dict]): 包含每个图像预处理时使用的变换参数的字典列表。
                                           每个字典应包含 'scale' (缩放比例) 和 'padding' (左上角填充量)。
            input_shape (tuple[int, int]): 模型输入的图像尺寸，格式为 (height, width)。
                                           用于反归一化步骤。
            box_format (str, optional): 输入边界框的格式。可选 'xyxy' 或 'cxcywh'。默认为 'xyxy'。

        Returns:
            np.ndarray: 转换回原始图像坐标系且为 'xyxy' 格式的边界框，形状与 `boxes` 相同。

        Raises:
            ValueError: 如果 `box_format` 不是 'xyxy' 或 'cxcywh'。

        """
        result = boxes.copy()
        batch_size = result.shape[0]
        model_h, model_w = input_shape

        # --- 调试：打印第一个有效的框来看看 ---
        # print(f"DEBUG Raw Box sample: {result[0, 0]}")

        # 1. 自动判断是否需要反归一化 (Normalize Check)
        # 如果最大值 <= 1.5，说明是 0-1 归一化数据，需要乘上 Input Size
        if result.size > 0 and result.max() <= 1.5:
            result[..., 0] *= model_w
            result[..., 2] *= model_w
            result[..., 1] *= model_h
            result[..., 3] *= model_h

        # 2. 格式转换逻辑
        # 只有当它是 cxcywh 时，才需要转换成 xyxy
        # 如果模型输出本身就是 xyxy，则跳过此步
        if box_format == 'cxcywh':
            cx, cy, w, h = result[..., 0], result[..., 1], result[..., 2], result[..., 3]
            result[..., 0] = cx - 0.5 * w  # x1
            result[..., 1] = cy - 0.5 * h  # y1
            result[..., 2] = cx + 0.5 * w  # x2
            result[..., 3] = cy + 0.5 * h  # y2

        # 3. 还原到原图 (Padding & Scale)
        for i in range(batch_size):
            params = transform_params[i]
            scale = params['scale']
            pad_left, pad_top = params['padding']

            # 减去 Padding
            result[i, :, 0] -= pad_left  # x1
            result[i, :, 1] -= pad_top  # y1
            result[i, :, 2] -= pad_left  # x2
            result[i, :, 3] -= pad_top  # y2

            # 除以缩放比例
            result[i, :, :4] /= scale

        return result

    @staticmethod
    def restore_masks(masks: np.ndarray, transform_params: list[dict],input_shape: tuple[int, int]) -> list[np.ndarray]:
        """将模型输出的 Mask 还原到原始图像尺寸。（input_shape为缩放后用于模型输入的图像尺寸，将会还原到原始图片尺寸）

        此方法处理以下步骤：
        1. **维度处理**: 确保 Mask 数组的维度正确，如果输入是 (Num_Queries, Mask_H, Mask_W)，则添加 Batch 维度。
        2. **裁剪 Mask**: 根据预处理时 Letterbox 操作引入的 padding，从 Mask 中裁剪出有效区域。
        3. **缩放 Mask**: 将裁剪后的 Mask 缩放回原始图像的尺寸。
        4. **二值化**: 将缩放后的 Mask 转换为二值 Mask (0 或 255)。

        Args:
            masks (np.ndarray): 模型输出的 Mask 数组。
                                期望形状为 `(Batch, Num_Queries, Mask_H, Mask_W)`
                                或 `(Num_Queries, Mask_H, Mask_W)` (如果 Batch 为 1)。
                                Mask 值通常是 logits。
            transform_params (list[dict]): 包含每个图像预处理时使用的变换参数的字典列表。
                                           每个字典应包含 'orig_shape' (原始图像形状) 和 'padding' (左上角填充量)。
            input_shape (tuple[int, int]): 模型输入的图像尺寸，格式为 (height, width)。
                                           例如，如果模型输入是 1024x1024，则为 (1024, 1024)。

        Returns:
            list[np.ndarray]: 还原到原始图像尺寸的二值 Mask 列表。
                              列表中的每个元素是一个 NumPy 数组，形状为 `(Num_Queries, Orig_H, Orig_W)`，
                              像素值为 0 或 255。

        Raises:
            ValueError: 如果 `masks` 的维度不符合预期。
        """
        input_h, input_w = input_shape

        restored_results = []
        if masks.ndim == 3:
            masks = masks[None, ...]

        # 获取模型输出的 mask 尺寸 (288, 288)
        mask_h, mask_w = masks.shape[-2:]


        for i, mask_tensor in enumerate(masks):
            params = transform_params[i]
            orig_h, orig_w = params['orig_shape']

            # 预处理时的 padding (针对 1008x1008 的)
            pad_w, pad_h = params['padding']  # (left, top)

            # 计算比例因子: 288 / 1008
            scale_y = mask_h / input_h
            scale_x = mask_w / input_w

            # 将 padding 映射到 mask 尺寸上
            mask_pad_w = pad_w * scale_x
            mask_pad_h = pad_h * scale_y

            # 计算 mask 上的有效区域
            valid_h = mask_h - (mask_pad_h * 2)
            valid_w = mask_w - (mask_pad_w * 2)

            # 裁剪 Mask (去除 Padding)
            # 注意坐标取整
            x1 = int(round(mask_pad_w))
            y1 = int(round(mask_pad_h))
            x2 = int(round(mask_pad_w + valid_w))
            y2 = int(round(mask_pad_h + valid_h))

            # 保护边界
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(mask_w, x2), min(mask_h, y2)

            cropped_masks = mask_tensor[:, y1:y2, x1:x2]  # (200, h, w)

            # 还原: Resize 回原图尺寸 (Orig_H, Orig_W)
            # OpenCV resize 需要 (W, H)
            curr_img_masks = []


            if cropped_masks.shape[0] > 0:
                # 转置为 (H, W, N) 以便通过 cv2 一次性 resize
                cropped_masks_trans = cropped_masks.transpose(1, 2, 0)

                # Resize
                # 注意：SAM 输出的是 logits，线性插值比较好
                restored_trans = cv2.resize(cropped_masks_trans, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

                # 如果只有一个 mask，resize 结果会掉维度 (H, W)，需要补回
                if restored_trans.ndim == 2:
                    restored_trans = restored_trans[..., None]

                # 转回 (N, H, W)
                restored = restored_trans.transpose(2, 0, 1)

                # 二值化 (Logits > 0)
                curr_img_masks = (restored > 0).astype(np.uint8) * 255

            restored_results.append(curr_img_masks)

        return restored_results
