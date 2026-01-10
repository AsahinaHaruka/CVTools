"""
Author: Haruka
Date: 2026-01-09 08:53:47
LastEditors: Haruka
LastEditTime: 2026-01-09 10:58:10
FilePath: /road/inference/sam3_inference.py
"""

import numpy as np

from .base_inference import ONNXInference
from .image_processor import ImageProcessor
from .tokenizer import SimpleCLIPBPETokenizer


class SAM3Inference(ONNXInference):
    def __init__(self,
                 model_path: str,
                 vocab_file: str,
                 merges_file: str,
                 enable_trt_profile: bool = True,
                 max_batch_size: int = 1,
                 opt_batch_size: int = 1,
                 input_image_size: tuple[int, int] = None,
                 target_long_side: int = 640,
                 other_size=(32, 32, 32)):
        """
        Initialize the ONNX inference session.
        :param model_path: Path to the ONNX model file.
        :param enable_trt_profile: 是否启用 TensorRT 固定形状优化
        :param max_batch_size: 预期的最大 Batch Size (仅在 enable_trt_profile=True 且模型输入动态时生效)
        :param opt_batch_size: 预期的最优 Batch Size (仅在 enable_trt_profile=True 且模型输入动态时生效)
        :param input_image_size: 摄像头/图片的原始分辨率 (Width, Height)。
                                 如果传入此参数，会自动计算最佳的矩形输入尺寸 (Rectangular Inference)。
                                 如果不传，默认使用 target_long_side 的正方形。
        :param target_long_side: 预期的输入尺寸 (H, W) (仅在 enable_trt_profile=True 且模型输入动态时生效)
        """
        super().__init__(model_path=model_path,
                         stride=14,
                         enable_trt_profile=enable_trt_profile,
                         max_batch_size=max_batch_size,
                         opt_batch_size=opt_batch_size,
                         input_image_size=input_image_size,
                         target_long_side=target_long_side,
                         other_size=other_size
                         )

        self.tokenizer = SimpleCLIPBPETokenizer(vocab_file=vocab_file,
                                                merges_file=merges_file,
                                                max_length=32,
                                                bos_token_id=49406,
                                                eos_token_id=49407,
                                                bpe_vocab_size=49152,
                                                )

        self.image_processor = ImageProcessor(target_size=self.img_size,
                                              stride=self.stride,
                                              is_fixed_size=self.fix_image,
                                              norm_type='-1_1',
                                              fill_value=144,
                                              dtype=self.input_meta[0]['type'])

    def __call__(self, input_data: dict[str, np.ndarray] | list[np.ndarray] | np.ndarray, prompt: str,
                 return_boxes: bool = True, return_masks: bool = True, raw: bool = False) -> dict:
        ids, mask = self.tokenizer.encode(prompt)
        input_ids = np.array(ids, dtype=np.int64).reshape(1, -1)
        attention_mask = np.array(mask, dtype=np.int64).reshape(1, -1)

        processed_input, transform_params = self.image_processor(input_data)

        outputs = super().__call__({
            "pixel_values": processed_input,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })

        res = {'scores': outputs[2]}
        if return_boxes:
            res['boxes'] = outputs[1] if raw else self.image_processor.convert_normalized_boxes(outputs[1],
                                                                                                transform_params,
                                                                                                input_shape=(
                                                                                                    self.image_processor.target_h,
                                                                                                    self.image_processor.target_w),
                                                                                                box_format='xyxy')
        if return_masks:
            res['masks'] = outputs[0] if raw else self.image_processor.restore_masks(outputs[0],
                                                                                     transform_params,
                                                                                     input_shape=(
                                                                                         self.image_processor.target_h,
                                                                                         self.image_processor.target_w))
        return res
