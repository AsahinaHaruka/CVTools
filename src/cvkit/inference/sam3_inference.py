"""
@Project:  CVKit
@File:     sam3_inference.py
@Author:   Haruka
@Date:     2026/01/09 08:53:47
"""

import numpy as np

from .base_inference import ONNXInference
from .image_processor import ImageProcessor
from .tokenizer import SimpleCLIPBPETokenizer


class SAM3Inference:
    def __init__(
        self,
        model_path: str,
        vocab_file: str,
        merges_file: str,
        enable_trt_profile: bool = False,
        stride: int = 14,
        min_batch_size: int = 1,
        opt_batch_size: int = 1,
        max_batch_size: int = 1,
        input_image_size: tuple[int, int] | None = None,
        target_long_side: int = 640,
        other_size=(32, 32, 32),
        execution_provider: tuple[str] = ("trt", "cuda", "CoreML", "cpu"),
    ):
        """
        初始化 SAM3 推理会话。

        加载 SAM3 ONNX 模型并配置推理会话选项。支持 TensorRT 执行提供程序及其动态形状配置。
        如果启用了 TensorRT 配置文件 (`enable_trt_profile=True`)，将根据提供的批处理大小和图像尺寸参数构建优化配置文件。

        Args:
            model_path (str): ONNX 模型文件的路径。
            vocab_file (str): CLIP BPE tokenizer 的词汇表文件路径。
            merges_file (str): CLIP BPE tokenizer 的合并文件路径。
            enable_trt_profile (bool, optional): 是否启用 TensorRT 动态形状固定化优化。
                如果为 True，将根据 batch size 和尺寸参数预热 TensorRT 引擎缓存，
                当此参数为 True 时，无论是否提供 `input_image_size`，`fix_image` 都将被设置为 False。
                默认为 True。
            stride (int, optional): 模型步长，用于计算对齐后的输入尺寸。默认为 14。
            min_batch_size (int, optional): 预期的最小 Batch Size。
                仅在 `enable_trt_profile=True` 且模型输入包含动态 Batch 维度时生效。默认为 1。
            opt_batch_size (int, optional): 预期的最优 Batch Size。
                仅在 `enable_trt_profile=True` 且模型输入包含动态 Batch 维度时生效。默认为 1。
            max_batch_size (int, optional): 预期的最大 Batch Size。
                仅在 `enable_trt_profile=True` 且模型输入包含动态 Batch 维度时生效。默认为 1。
            input_image_size (tuple[int, int] | None, optional): 原始输入图像的分辨率 (Width, Height)。
                如果提供，将基于 `target_long_side` 计算最佳矩形推理尺寸 (Rectangular Inference)，
                此时 `fix_image` 将被设置为 True。
                如果不提供，默认使用 `target_long_side` 作为边长的正方形尺寸，此时 `fix_image` 为 False。
                默认为 None。
            target_long_side (int, optional): 预期的输入图像长边尺寸。
                用于计算实际推理时的输入分辨率。
                默认为 640。
            other_size (tuple[int, int, int], optional): 非图像维度的动态输入尺寸配置 (min, opt, max)。
                用于 TensorRT Profile 中非 Batch、非 Image (H/W) 的动态维度（如序列长度）。
                默认为 (32, 32, 32)。
            execution_provider (tuple[str], optional): 需要扫描的后端列表

        Raises:
            FileNotFoundError: 如果 `model_path` 指定的文件不存在。
        """
        self.model = ONNXInference(
            model_path=model_path,
            stride=stride,
            enable_trt_profile=enable_trt_profile,
            min_batch_size=min_batch_size,
            opt_batch_size=opt_batch_size,
            max_batch_size=max_batch_size,
            input_image_size=input_image_size,
            target_long_side=target_long_side,
            other_size=other_size,
            execution_provider=execution_provider,
        )

        self.tokenizer = SimpleCLIPBPETokenizer(
            vocab_file=vocab_file,
            merges_file=merges_file,
            max_length=32,
            bos_token_id=49406,
            eos_token_id=49407,
            bpe_vocab_size=49152,
        )

        self.image_processor = ImageProcessor(
            target_size=self.model.img_size,
            stride=self.model.stride,
            is_fixed_size=self.model.fix_image,
            norm_type="-1_1",
            fill_value=144,
            dtype=self.model.input_meta[0]["type"],
        )

    def __call__(
        self,
        input_data: dict[str, np.ndarray] | list[np.ndarray] | np.ndarray,
        prompt: str,
        return_boxes: bool = True,
        return_masks: bool = True,
        raw: bool = False,
    ) -> dict:
        """
        执行 SAM3 模型的推理。

        该方法首先对输入的文本提示进行编码，然后对图像数据进行预处理。
        接着，将处理后的图像和文本输入模型进行推理，并根据 `return_boxes` 和 `return_masks` 参数
        选择性地对输出的边界框和掩码进行后处理（还原到原始图像坐标系）。

        Args:
            input_data (dict[str, np.ndarray] | list[np.ndarray] | np.ndarray):
                输入图像数据。可以是单个 NumPy 数组 (H, W) 或 (H, W, C)，
                也可以是包含多个 NumPy 数组的列表，或者是一个字典（如果模型有多个图像输入）。
            prompt (str): 用于指导 SAM3 模型生成分割结果的文本提示。
            return_boxes (bool, optional): 是否在结果中包含边界框。默认为 True。
            return_masks (bool, optional): 是否在结果中包含分割掩码。默认为 True。
            raw (bool, optional): 如果为 True，则返回模型原始输出的边界框和掩码（未经过后处理）；
                                  否则，边界框和掩码将被还原到原始图像坐标系。默认为 False。

        Returns:
            dict: 包含推理结果的字典。可能包含以下键：
                - 'scores' (np.ndarray): 模型的置信度分数。
                - 'boxes' (np.ndarray, optional): 边界框坐标，格式为 'xyxy'。
                                                  如果 `raw` 为 True，则为模型输出坐标；
                                                  否则为原始图像坐标。
                - 'masks' (list[np.ndarray], optional): 分割掩码列表。
                                                       如果 `raw` 为 True，则为模型输出掩码；
                                                       否则为还原到原始图像尺寸的二值掩码(0/255)。

        Raises:
            ValueError: 如果 `input_data` 格式不符合预期或模型输入数量不匹配。
        """
        ids, mask = self.tokenizer.encode(prompt)
        input_ids = np.array(ids, dtype=np.int64).reshape(1, -1)
        attention_mask = np.array(mask, dtype=np.int64).reshape(1, -1)

        processed_input, transform_params = self.image_processor(input_data)

        outputs = self.model(
            {
                "pixel_values": processed_input,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )

        res = {"scores": outputs[2]}
        if return_boxes:
            res["boxes"] = (
                outputs[1]
                if raw
                else self.image_processor.restore_boxes(
                    outputs[1], transform_params, box_format="xyxy"
                )
            )
        if return_masks:
            res["masks"] = (
                outputs[0]
                if raw
                else self.image_processor.restore_masks(outputs[0], transform_params)
            )
        return res
