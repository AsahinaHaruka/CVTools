"""
@Project : CVTools 
@File : image_processor.py
@Author : Haruka
@Date : 2026/1/9 13:41 
"""
import cv2
import numpy as np


def zero2one(input_tensor: np.ndarray) -> np.ndarray:
    return input_tensor.astype(np.float32) / 255.0


def minus_one2one(input_tensor: np.ndarray) -> np.ndarray:
    input_tensor = input_tensor.astype(np.float32) / 255.0
    return (input_tensor - 0.5) / 0.5


def imagenet(input_tensor: np.ndarray) -> np.ndarray:
    input_tensor = input_tensor.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
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
        图像预处理与后处理工具类

        :param target_size: 目标尺寸 (int or (h, w))
        :param stride: 模型步长，用于计算 padding 对齐
        :param is_fixed_size: 是否强制缩放到固定尺寸 (TensorRT/Static Shape 常用)
        :param fill_value: Padding 的填充值
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
        预处理输入数据
        :param input_data: 输入图像列表
        :return: 预处理后的numpy数组
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

    @staticmethod
    def convert_normalized_boxes(boxes: np.ndarray,
                                 transform_params: list,
                                 input_shape: tuple[int, int],
                                 box_format: str = 'xyxy') -> np.ndarray:
        """
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
        """
        还原 Mask (处理输出尺寸与输入尺寸不一致的情况)
        :param masks: (Batch, Num_Queries, Mask_H, Mask_W) -> (1, 200, 288, 288)
        :param transform_params: 预处理参数
        """
        input_h, input_w = input_shape
        restored_results = []
        if masks.ndim == 3:
            masks = masks[None, ...]

        # 获取模型输出的 mask 尺寸 (288, 288)
        mask_h, mask_w = masks.shape[-2:]

        # 获取模型输入的尺寸 (1008, 1008) - 从 self 中获取或者假设是正方形
        # transform_params 中的 scale 是相对于 input_size (1008) 的
        # 我们需要计算 mask 相对于 input_size 的缩放因子

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
