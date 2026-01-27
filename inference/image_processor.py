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


NORM_CALL = {
    '0_1': zero2one,
    '-1_1': minus_one2one,
    'imagenet': imagenet
}


class ImageProcessor:
    def __init__(self,
                 target_size: int | tuple[int, int] = 640,
                 stride: int = 32,
                 is_fixed_size: bool = False,
                 norm_type: str = "0_1",
                 fill_value: int = 114,
                 dtype: np.dtype = np.float32,
                 uniform_transform: bool = False):
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
            uniform_transform (bool): 如果为 True，假设 Batch 内所有图片
                                  使用了完全相同的预处理（如视频流）。
                                  这将启用矩阵运算加速，不再循环处理。
        """
        self.stride = stride
        self.is_fixed_size = is_fixed_size
        self.fill_value = fill_value
        self.dtype = dtype

        self._norm_function = NORM_CALL.get(norm_type)

        # 统一处理 target_size
        if isinstance(target_size, int):
            self.target_h, self.target_w = target_size, target_size
            self.long_side = target_size
        else:
            self.target_h, self.target_w = target_size
            self.long_side = max(self.target_h, self.target_w)

        self.uniform_transform = uniform_transform

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
            'padding': (left, top),
            'input_shape': transformed_img.shape[:2]
        }
        return transformed_img, transform_params

    def _batch_letterbox(self, images: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        针对 Batch 图片的 Letterbox 处理。
        Images 形状必须为 (B, H, W, C) 且所有图片原始尺寸一致。
        """
        batch_size, h_orig, w_orig, c = images.shape

        # 1. 计算统一的变换参数 (只计算一次)
        if self.is_fixed_size:
            r = min(self.target_h / h_orig, self.target_w / w_orig)
            new_w, new_h = int(round(w_orig * r)), int(round(h_orig * r))
            dw, dh = self.target_w - new_w, self.target_h - new_h
        else:
            r = self.long_side / max(h_orig, w_orig)
            new_w, new_h = int(round(w_orig * r)), int(round(h_orig * r))
            dw, dh = self.long_side - new_w, self.long_side - new_h
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)

        dw /= 2
        dh /= 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        # 2. 批量 Resize (Vertical Stacking Trick)
        # 将 (B, H, W, C) -> (B*H, W, C) 变成一张长图
        if (h_orig, w_orig) != (new_h, new_w):
            stacked_img = images.reshape(batch_size * h_orig, w_orig, c)
            # Resize 到 (new_w, B * new_h)
            # 注意: cv2.resize 接收 (Width, Height)
            resized_stacked = cv2.resize(stacked_img, (new_w, batch_size * new_h), interpolation=cv2.INTER_LINEAR)
            # 还原维度 (B, new_h, new_w, C)
            resized_batch = resized_stacked.reshape(batch_size, new_h, new_w, c)
        else:
            resized_batch = images

        # 3. 批量 Padding (利用 numpy 广播赋值)
        final_h = new_h + top + bottom
        final_w = new_w + left + right

        # 预分配大数组
        padded_batch = np.full((batch_size, final_h, final_w, c),
                               self.fill_value, dtype=images.dtype)

        # 一次性填入中间区域
        padded_batch[:, top:top + new_h, left:left + new_w, :] = resized_batch

        # 4. 构造参数列表 (所有图片参数一致)
        common_params = {
            'orig_shape': (h_orig, w_orig),
            'scale': r,
            'padding': (left, top),
            'input_shape': (final_h, final_w)
        }
        transform_params = [common_params] * batch_size

        return padded_batch, transform_params

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
        if isinstance(input_data, np.ndarray):
            # (H, W) or (H, W, C) -> 单张图，包装成 list 或 扩展维度
            if input_data.ndim == 2 or input_data.ndim == 3:
                input_data = [input_data]

            # --- 2. 逻辑分支 ---
        if self.uniform_transform:
            # ===  (Batch Matrix Operation) ===
            # 所有输入图片尺寸一致

            # A. 准备数据: List -> Numpy Batch
            if isinstance(input_data, list):
                # 如果是 list，必须 stack 起来。
                # 如果图片尺寸不一致，这里 np.stack 会直接报错，符合预期 (Fail Fast)
                if len(input_data) == 0:
                    return np.empty((0, 3, self.target_h, self.target_w), dtype=self.dtype), []
                input_tensor = np.stack(input_data)
            else:
                input_tensor = input_data

            # B. 维度修正: (B, H, W) -> (B, H, W, 1) 处理灰度图
            if input_tensor.ndim == 3:
                input_tensor = input_tensor[..., None]

            # 批量处理
            transformed_tensor, transform_params = self._batch_letterbox(input_tensor)

        else:
            # 尺寸不一的图片列表
            transformed_data = []
            transform_params = []
            # 此时 input_data 应该是一个 list 或者 iterable 的 numpy array
            for img in input_data:
                t_img, t_params = self.letterbox(img)
                transformed_data.append(t_img)
                transform_params.append(t_params)

            transformed_tensor = np.stack(transformed_data)

            # --- 通用后处理 (HWC -> CHW, BGR -> RGB, Normalize) ---
            # 此时 transformed_tensor 形状为 (B, H, W, C)

            # BGR -> RGB
        if transformed_tensor.shape[-1] == 3:
            transformed_tensor = transformed_tensor[..., ::-1]

            # BHWC -> BCHW
        transformed_tensor = transformed_tensor.transpose((0, 3, 1, 2))
        # 归一化
        transformed_tensor = self._norm_function(transformed_tensor)

        return transformed_tensor.astype(self.dtype), transform_params

    @staticmethod
    def restore_boxes(detections: np.ndarray,
                      transform_params: list[dict],
                      box_format: str = 'xyxy') -> np.ndarray:
        """
        将检测框坐标还原到原始图像坐标系（通用版）。
        Args:
            detections (np.ndarray): 模型的输出结果，形状为 (Batch, Num_Queries, D)。
                D >= 4，前 4 列必须是坐标 (x, y, x, y) 或 (cx, cy, w, h)。
                如果是单纯的 boxes，D=4；如果是包含置信度的结果，D>4，该函数只修改前 4 列。
            transform_params (list[dict]): 包含每个图像预处理参数的列表 (scale, padding)。
            box_format (str): 输入坐标的格式，'xyxy' 或 'cxcywh'。默认为 'xyxy'。

        Returns:
            np.ndarray: 还原后的结果，形状与输入相同。坐标部分已转换为原始图像坐标且格式统一为 'xyxy'。
        """
        # 避免修改原数据
        result = detections.copy()

        # 获取 batch size
        batch_size = result.shape[0]

        # ---------------------------------------------------------
        # 1. 自动反归一化 (Normalize Check)
        # ---------------------------------------------------------
        # 检查前4列的最大值。如果 <= 1.5 且提供了 input_shape，说明是 0-1 归一化数据
        input_shape = transform_params[0].get('input_shape')
        if input_shape is not None and result[..., :4].max() <= 1.5:
            h, w = input_shape
            result[..., 0] *= w  # x or cx
            result[..., 1] *= h  # y or cy
            result[..., 2] *= w  # x or w
            result[..., 3] *= h  # y or h

        # ---------------------------------------------------------
        # 2. 格式转换 (cxcywh -> xyxy)
        # ---------------------------------------------------------
        if box_format == 'cxcywh':
            cx, cy, w, h = result[..., 0], result[..., 1], result[..., 2], result[..., 3]
            # 这里使用临时变量，避免原地修改导致计算错误
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h

            result[..., 0], result[..., 1], result[..., 2], result[..., 3] = x1, y1, x2, y2

        # ---------------------------------------------------------
        # 3. 还原 Letterbox (去除 Padding 并除以 Scale)
        # ---------------------------------------------------------
        # 向量化处理：提取 scale 和 padding
        scales = np.array([p['scale'] for p in transform_params], dtype=np.float32)[:, None, None]  # (B, 1, 1)
        pads = np.array([p['padding'] for p in transform_params], dtype=np.float32)  # (B, 2)
        pad_left = pads[:, 0][:, None]  # (B, 1)
        pad_top = pads[:, 1][:, None]  # (B, 1)

        # 减去 padding
        result[..., 0] -= pad_left
        result[..., 2] -= pad_left
        result[..., 1] -= pad_top
        result[..., 3] -= pad_top

        # 除以 scale
        result[..., :4] /= scales

        return result

    @staticmethod
    def restore_masks(masks: np.ndarray | list[np.ndarray],
                      transform_params: list[dict],
                      boxes: np.ndarray | None = None,
                      mask_threshold: float = 0.0,
                      uniform_transform: bool = False) -> list[np.ndarray]:
        """将模型输出的 Mask 还原到原始图像尺寸。（input_shape为缩放后用于模型输入的图像尺寸，将会还原到原始图片尺寸）

        此方法处理以下步骤：
        1. 维度处理: 确保 Mask 数组的维度正确。
        2. (可选) Crop to Box: 如果提供了 `boxes`，则将 Mask 裁剪到对应的边界框范围内。
        3. 裁剪 Mask: 根据预处理时 Letterbox 操作引入的 padding，从 Mask 中裁剪出有效区域。
        4. 缩放 Mask: 将裁剪后的 Mask 缩放回原始图像的尺寸。
        5. 二值化: 将缩放后的 Mask 转换为二值 Mask (0 或 255)。

        Args:
            masks (np.ndarray | list[np.ndarray]): 模型输出的 Mask 数组。
                                期望形状为 `(Batch, Num_Queries, Mask_H, Mask_W)`
                                或 `(Num_Queries, Mask_H, Mask_W)` (如果 Batch 为 1)。
                                也可以是 list[np.ndarray]，每个元素为 (Num_Queries, Mask_H, Mask_W)。
                                Mask 值通常是 logits。
            transform_params (list[dict]): 包含每个图像预处理时使用的变换参数的字典列表。
                                           每个字典应包含 'orig_shape', 'padding', 'input_shape'。
            boxes (np.ndarray, optional): 对应的边界框，用于裁剪 Mask。
                                       形状应为 `(Batch, Num_Queries, 4)` 或 `(Num_Queries, 4)`。
                                       坐标必须是模型输出坐标系（即未还原的坐标），格式为 xyxy。
                                       如果提供，将执行 Crop to Box 操作，去除框外的噪声。
            mask_threshold (float): Mask 二值化的阈值。默认为 0.0 (适用于 Logits)。
                                    如果是概率值 (0-1)，建议设置为 0.5。

            uniform_transform (bool): 如果为 True，假设 Batch 内所有图片
                                    使用了完全相同的预处理（如视频流）。
                                    这将启用矩阵运算加速，不再循环处理。

        Returns:
            list[np.ndarray]: 还原到原始图像尺寸的二值 Mask 列表。
                              列表中的每个元素是一个 NumPy 数组，形状为 `(Num_Queries, Orig_H, Orig_W)`，
                              像素值为 0 或 255。

        Raises:
            ValueError: 如果 `masks` 的维度不符合预期。
        """
        # --- 1. 数据标准化处理 ---
        # 确保 masks 是 (B, N, H, W)
        if isinstance(masks, np.ndarray):
            if masks.ndim == 3:
                masks = masks[None, ...]
        elif isinstance(masks, list):
            if len(masks) == 0:
                return []
            # 如果启用加速，必须转换为 numpy 数组
            if uniform_transform:
                masks = np.array(masks)
        else:
            raise ValueError("masks must be np.ndarray or list[np.ndarray]")

        # 确保 boxes 是 (B, N, 4)
        if boxes is not None:
            if boxes.ndim == 2:
                boxes = boxes[None, ...]

        restored_results = []

        # --- 2. 向量化 Crop to Box (去除框外噪声) ---
        if isinstance(masks, np.ndarray) and boxes is not None:
            # batch 内 input_shape 大致一致（用于计算相对比例），取第一个即可
            input_h, input_w = transform_params[0]['input_shape']
            mask_h, mask_w = masks.shape[-2:]

            scale_y = mask_h / input_h
            scale_x = mask_w / input_w

            # boxes: (B, N, 4) -> 扩展维度用于广播
            b_x1 = (boxes[..., 0] * scale_x)[..., None, None]
            b_y1 = (boxes[..., 1] * scale_y)[..., None, None]
            b_x2 = (boxes[..., 2] * scale_x)[..., None, None]
            b_y2 = (boxes[..., 3] * scale_y)[..., None, None]

            # 生成网格
            x_range = np.arange(mask_w, dtype=np.float32)[None, None, None, :]
            y_range = np.arange(mask_h, dtype=np.float32)[None, None, :, None]

            # 计算掩码并应用 (In-place multiplication to save memory if possible)
            crop_mask = (x_range >= b_x1) & (x_range < b_x2) & (y_range >= b_y1) & (y_range < b_y2)
            masks = masks * crop_mask

        if uniform_transform and isinstance(masks, np.ndarray):
            # 假设所有图片参数一致，取第一个
            params = transform_params[0]
            orig_h, orig_w = params['orig_shape']
            input_h, input_w = params['input_shape']
            pad_w, pad_h = params['padding']

            batch_size, num_queries, mask_h, mask_w = masks.shape

            # 计算裁剪区域 (Remove Padding)
            scale_y = mask_h / input_h
            scale_x = mask_w / input_w

            mask_pad_w = pad_w * scale_x
            mask_pad_h = pad_h * scale_y

            valid_h = mask_h - (mask_pad_h * 2)
            valid_w = mask_w - (mask_pad_w * 2)

            x1 = max(0, int(round(mask_pad_w)))
            y1 = max(0, int(round(mask_pad_h)))
            x2 = min(mask_w, int(round(mask_pad_w + valid_w)))
            y2 = min(mask_h, int(round(mask_pad_h + valid_h)))

            # 批量切片: (B, N, Valid_H, Valid_W)
            cropped_masks = masks[..., y1:y2, x1:x2]

            if cropped_masks.size == 0:
                return [np.zeros((num_queries, orig_h, orig_w), dtype=np.uint8) for _ in range(batch_size)]

            # 筛选有效 Mask
            mask_max_values = cropped_masks.max(axis=(-2, -1))
            valid_indices = mask_max_values > mask_threshold

            # 获取有效 Mask 的数据 (Flatten Batch & Query dims)
            valid_crops = cropped_masks[valid_indices]

            # 预分配最终结果容器
            # 形状: (Total_B_N, Orig_H, Orig_W), 类型 uint8, 初始全 0
            total_masks_count = batch_size * num_queries
            final_output_flat = np.zeros((total_masks_count, orig_h, orig_w), dtype=np.uint8)

            num_valid = valid_crops.shape[0]
            if num_valid > 0:
                mask_binary_valid = (valid_crops > mask_threshold).astype(np.uint8) * 255

                flat_indices_map = np.flatnonzero(valid_indices.reshape(-1))

                chunk_size = 16
                for i in range(0, num_valid, chunk_size):
                    # 当前块的结束位置
                    end = min(i + chunk_size, num_valid)
                    # 取出当前块数据 (C, H, W)
                    chunk = mask_binary_valid[i: end]  # (C, H, W)
                    # Transpose: (H, W, C)
                    to_resize = chunk.transpose(1, 2, 0)

                    # Resize
                    resized_chunk = cv2.resize(to_resize, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

                    # 修正维度丢失问题 (如果 Chunk Size 为 1)
                    if resized_chunk.ndim == 2:
                        resized_chunk = resized_chunk[..., None]

                    # 再次二值化，消除插值产生的非 0/255 值
                    _, resized_chunk_bin = cv2.threshold(resized_chunk, 127, 255, cv2.THRESH_BINARY)
                    current_indices = flat_indices_map[i: end]
                    final_output_flat[current_indices] = resized_chunk_bin.transpose(2, 0, 1)

            return list(final_output_flat.reshape(batch_size, num_queries, orig_h, orig_w))

        else:
            # 逐个循环
            for i, mask_tensor in enumerate(masks):
                params = transform_params[i]
                input_h, input_w = params['input_shape']
                mask_h, mask_w = mask_tensor.shape[-2:]
                orig_h, orig_w = params['orig_shape']
                pad_w, pad_h = params['padding']

                # 计算比例
                scale_y = mask_h / input_h
                scale_x = mask_w / input_w

                # 计算有效区域
                mask_pad_w = pad_w * scale_x
                mask_pad_h = pad_h * scale_y
                valid_h = mask_h - (mask_pad_h * 2)
                valid_w = mask_w - (mask_pad_w * 2)

                x1 = max(0, int(round(mask_pad_w)))
                y1 = max(0, int(round(mask_pad_h)))
                x2 = min(mask_w, int(round(mask_pad_w + valid_w)))
                y2 = min(mask_h, int(round(mask_pad_h + valid_h)))

                # Crop
                cropped_masks = mask_tensor[:, y1:y2, x1:x2]

                # Resize
                curr_img_masks = np.zeros((mask_tensor.shape[0], orig_h, orig_w), dtype=np.uint8)

                if cropped_masks.shape[1] > 0 and cropped_masks.shape[2] > 0:
                    # 找出哪些 query 是有效的
                    valid_idx = cropped_masks.max(axis=(1, 2)) > mask_threshold
                    # 如果存在有效的 mask
                    if np.any(valid_idx):
                        # 只处理有效的
                        valid_crops = cropped_masks[valid_idx]  # (Valid_N, H, W)

                        # Transpose (N, H, W) -> (H, W, N)
                        valid_crops_trans = valid_crops.transpose(1, 2, 0).astype(np.float32)

                        # Resize
                        restored_trans = cv2.resize(valid_crops_trans, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

                        if restored_trans.ndim == 2:
                            restored_trans = restored_trans[..., None]

                        # 二值化
                        restored_valid = (restored_trans > mask_threshold).astype(np.uint8) * 255

                        # 填回容器 (Scatter)
                        # restored_valid shape is (H, W, Valid_N) -> transpose to (Valid_N, H, W)
                        curr_img_masks[valid_idx] = restored_valid.transpose(2, 0, 1)

                restored_results.append(curr_img_masks)

            return restored_results
