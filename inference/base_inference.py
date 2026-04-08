"""
@Project : CVTools 
@File : base_inference.py
@Author : Haruka
@Date : 2026/1/8 16:36 
"""
import math
import os
from typing import Sequence
import logging

import onnxruntime as ort
import numpy as np
from numpy import ndarray
from onnxruntime import SparseTensor

logger = logging.getLogger(__name__)

# 使用字典映射动态获取模型输入的数据类型
DTYPE_MAPPING = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(int8)": np.int8,
    "tensor(bool)": bool
}


class ONNXInference:
    def __init__(
            self,
            model_path: str,
            enable_trt_profile: bool = False,
            stride: int = 32,
            min_batch_size: int = 1,
            opt_batch_size: int = 5,
            max_batch_size: int = 5,
            input_image_size: tuple[int, int] | None = None,
            target_long_side: int = 640,
            other_size: tuple[int, int, int] = (0, 100, 200),
            execution_provider: tuple[str, ...] = ("trt", "cuda", "CoreML", "cpu"),
    ):
        """初始化 ONNX 推理会话。

        加载 ONNX 模型并配置推理会话选项。支持 TensorRT 执行提供程序及其动态形状配置。
        如果启用了 TensorRT 配置文件 (`enable_trt_profile=True`)，将根据提供的批处理大小和图像尺寸参数
        构建优化配置文件。

        Args:
            model_path (str): ONNX 模型文件的路径。
            enable_trt_profile (bool, optional): 是否启用 TensorRT 动态形状固定化优化。
                如果为 True，将根据 batch size 和尺寸参数预热 TensorRT 引擎缓存，
                该参数为True时无论是否提供input_image_size，fix_image为True。默认为 False。
            stride (int, optional): 模型步长，用于计算对齐后的输入尺寸。默认为 32。
            min_batch_size (int, optional): 预期的最小 Batch Size。
                仅在 `enable_trt_profile=True` 且模型输入包含动态 Batch 维度时生效。默认为 1。
            opt_batch_size (int, optional): 预期的最优 Batch Size。
                仅在 `enable_trt_profile=True` 且模型输入包含动态 Batch 维度时生效。默认为 5。
            max_batch_size (int, optional): 预期的最大 Batch Size。
                仅在 `enable_trt_profile=True` 且模型输入包含动态 Batch 维度时生效。默认为 5。
            input_image_size (tuple[int, int] | None, optional): 原始输入图像的分辨率 (Width, Height)。
                如果提供，将基于 `target_long_side` 计算最佳矩形推理尺寸 (Rectangular Inference)，此时fix_image为True。
                如果不提供，默认使用 `target_long_side` 作为边长的正方形尺寸，此时fix_image为False。默认为 None。
            target_long_side (int, optional): 预期的输入图像长边尺寸。
                用于计算实际推理时的输入分辨率。默认为 640。
            other_size (tuple[int, int, int], optional): 非图像维度的动态输入尺寸配置 (min, opt, max)。
                用于 TensorRT Profile 中非 Batch、非 Image (H/W) 的动态维度（如序列长度）。
                默认为 (0, 100, 200)。
            execution_provider (tuple[str, ...], optional): 需要扫描的后端执行提供者列表。
                默认为 ("trt", "cuda", "CoreML", "cpu")。

        Raises:
            FileNotFoundError: 如果 `model_path` 指定的文件不存在。
        """
        self.model_path = model_path
        self.stride = stride
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.opt_batch_size = opt_batch_size
        self.target_long_side = target_long_side

        self.fix_image = False

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, "rb") as f:
            model_bytes = f.read()

        # 预分析模型结构 有几个输入，哪些是图像，哪些是动态的
        self.input_meta, self.output_names = self._scan_model_io(model_bytes)

        fixed_h, fixed_w = self._check_static_shape()

        if fixed_h is not None and fixed_w is not None:
            #  如果模型尺寸固定，强制使用模型尺寸
            self.img_size = (fixed_h, fixed_w)
            self.fix_image = True

            logger.info(f"ℹ️ [Init] 检测到模型图像输入尺寸固定，强制使用固定图像输入: {self.img_size}")

        else:
            self.img_size = self._get_inference_size(
                self.target_long_side,
                stride,
                input_image_size
            )
            if self.fix_image:
                logger.info(
                    f"ℹ️ [Init] 使用固定推理尺寸。尺寸计算: 输入图片尺寸{input_image_size}，目标长边 {target_long_side} -> 计算推理尺寸 {self.img_size}")
            else:
                logger.info(f"ℹ️ [Init] 使用动态推理尺寸。目标长边 {target_long_side}")

        trt_provider_options = {
            'device_id': 0,
            'trt_max_workspace_size': 4 * 1024 * 1024 * 1024,  # 4GB
            'trt_fp16_enable': True,
        }

        if enable_trt_profile:
            trt_profile_options = self._build_trt_profile(other_size)
            self.fix_image = True

            if trt_profile_options:
                if trt_profile_options:
                    logger.info(f"ℹ️ [Init] 设置TensorRT 动态形状优化参数:\n"
                          f"   Min: {trt_profile_options['trt_profile_min_shapes']}\n"
                          f"   Opt: {trt_profile_options['trt_profile_opt_shapes']}\n"
                          f"   Max: {trt_profile_options['trt_profile_max_shapes']}")

                trt_provider_options.update(trt_profile_options)

            model_name = os.path.splitext(os.path.basename(model_path))[0]
            shape_tag = f"{opt_batch_size}_{max_batch_size}__{self.img_size[0]}x{self.img_size[1]}__{other_size[0]}_{other_size[1]}_{other_size[2]}"
            cache_dir = os.path.join("trt_cache", f"{model_name}__{shape_tag}")
            os.makedirs(cache_dir, exist_ok=True)

            trt_provider_options.update({
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": cache_dir,
            })

        available_providers = ort.get_available_providers()
        logger.debug(f"ℹ️ [Init] 系统当前可用后端: {available_providers}")
        providers = []

        # TensorRT
        if "trt" in execution_provider and "TensorrtExecutionProvider" in available_providers:
            # 只有当环境支持 TRT 时，才传入配置好的 trt_provider_options
            providers.append(("TensorrtExecutionProvider", trt_provider_options))

        #  CUDA
        if "cuda" in execution_provider and "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")

        # CoreML
        if "CoreML" in execution_provider and "CoreMLExecutionProvider" in available_providers:
            providers.append((
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "RequireStaticInputShapes": "1",
                    "EnableOnSubgraphs": "0",
                },
            )
            )

        # CPU
        if "cpu" in execution_provider:
            providers.append("CPUExecutionProvider")

        # 3. 创建正式的推理 Session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.log_severity_level = 3

        self.session = ort.InferenceSession(model_bytes, sess_options=session_options, providers=providers)
        logger.info(f"ℹ️ [Init] 模型加载成功! 运行设备: {self.session.get_providers()[0]}")

    @staticmethod
    def _scan_model_io(model_bytes: bytes):
        """扫描模型的输入输出元数据"""
        temp_sess = ort.InferenceSession(model_bytes, providers=['CPUExecutionProvider'])

        inputs = []
        for inp in temp_sess.get_inputs():
            inputs.append({
                'name': inp.name,
                'shape': inp.shape,
                'type': DTYPE_MAPPING.get(inp.type, np.float32)
            })
        logger.debug(f"Input Meta {inputs}")

        output_names = [out.name for out in temp_sess.get_outputs()]
        logger.debug(f"Output Names {output_names}")

        del temp_sess
        return inputs, output_names

    def _check_static_shape(self) -> tuple[int | None, int | None]:
        """
        检查模型是否具有固定的图像尺寸。
        假设：如果是 4D 输入 (N, C, H, W)，则检查 H(2) 和 W(3)。
        """
        for meta in self.input_meta:
            shape = meta['shape']
            # 如果是 4 维张量，通常认为是 NCHW 的图像输入
            if len(shape) == 4:
                h = shape[2]
                w = shape[3]
                # 在 ONNXRuntime 中，固定维度是 int，动态维度是 str 或 None
                if isinstance(h, int) and isinstance(w, int):
                    return h, w
        return None, None

    def _get_inference_size(self, target_long: int, stride: int, input_wh: tuple[int, int] = None) -> tuple[int, int]:
        """
        统一计算推理尺寸，并强制执行 Stride 对齐检查。

        该方法首先根据 `input_wh` 和 `target_long` 计算初步的缩放尺寸。
        然后，它将这些尺寸向上取整到 `stride` 的最近倍数，以确保模型输入尺寸的兼容性。

        Args:
            target_long (int): 目标长边尺寸。
            stride (int): 模型步长，用于对齐尺寸。
            input_wh (tuple[int, int] | None, optional): 原始输入图像的宽度和高度 (W, H)。
                如果提供，将根据原始图像比例和 target_long 计算推理尺寸。
                如果为 None，则默认使用 target_long 作为正方形的边长。

        Return: [Height, Width]
        """
        if input_wh is not None:
            # 矩形推理 (根据原图比例计算)
            self.fix_image = True
            w, h = input_wh
            scale = target_long / max(w, h)
            # 初步计算缩放后的尺寸
            raw_h = h * scale
            raw_w = w * scale
        else:
            # 正方形推理兜底
            raw_h = target_long
            raw_w = target_long

        # 执行对齐计算 (向上取整)
        new_h = int(math.ceil(raw_h / stride) * stride)
        new_w = int(math.ceil(raw_w / stride) * stride)

        # 计算预期的基础整数尺寸 (四舍五入)，用于和对齐后的尺寸对比
        expected_h = int(round(raw_h))
        expected_w = int(round(raw_w))

        if new_h != expected_h or new_w != expected_w:
            logger.warning(
                f"⚠️ [WARNING] 图片尺寸[{expected_h}, {expected_w}] 需要被 stride「{stride}」整除, 向上取整到[{new_h}, {new_w}]")

        return new_h, new_w

    def _build_trt_profile(self, other_size) -> dict:
        """
        生成支持多输入的 Profile 字符串。
        TRT 格式要求：name1:dim1 x dim2,name2:dim1 x dim2
        """

        min_profiles = []
        opt_profiles = []
        max_profiles = []

        for meta in self.input_meta:
            name = meta['name']
            shape = meta['shape']

            # 维度配置列表
            p_min, p_opt, p_max = [], [], []

            for idx, dim in enumerate(shape):
                if isinstance(dim, int):
                    # 静态维度，直接保留
                    p_min.append(dim)
                    p_opt.append(dim)
                    p_max.append(dim)
                elif isinstance(dim, str):
                    # 动态维度
                    if idx == 0:  # Batch Size
                        p_min.append(self.min_batch_size)
                        p_opt.append(self.opt_batch_size)
                        p_max.append(self.max_batch_size)
                    elif len(shape) == 4 and idx in [2, 3]:
                        # 只有当输入是 4D (NCHW) 时，才认为是图像，应用图像尺寸策略
                        # Height (2) / Width (3)
                        val = self.img_size[idx - 2]
                        p_min.append(val)
                        p_opt.append(val)
                        p_max.append(val)
                    else:
                        # 如果需要更精细控制，需要额外传参，这里给默认 1-100-200 策略或根据需求修改
                        p_min.append(other_size[0])
                        p_opt.append(other_size[1])
                        p_max.append(other_size[2])  # 非图像动态维度的上限

            # 将该输入的维度组合成字符串 NxCxHxW
            min_profiles.append(f"{name}:{'x'.join(map(str, p_min))}")
            opt_profiles.append(f"{name}:{'x'.join(map(str, p_opt))}")
            max_profiles.append(f"{name}:{'x'.join(map(str, p_max))}")

        return {
            'trt_profile_min_shapes': ','.join(min_profiles),
            'trt_profile_opt_shapes': ','.join(opt_profiles),
            'trt_profile_max_shapes': ','.join(max_profiles)
        }

    def __call__(self, input_data: dict[str, np.ndarray] | list[np.ndarray] | np.ndarray) -> Sequence[
        ndarray | SparseTensor | list | dict]:
        """执行 ONNX 模型的推理。

        此方法根据 `input_data` 的类型自动构建 `feed_dict`，然后执行 ONNX Runtime 会话。

        Args:
            input_data (dict[str, np.ndarray] | list[np.ndarray] | np.ndarray):
                输入数据，支持以下格式：
                - `dict[str, np.ndarray]`: 字典形式，键为模型输入名称，值为对应的 NumPy 数组。
                                          这是最推荐和最安全的方式，因为它明确指定了每个输入的名称。
                - `list[np.ndarray] | tuple[np.ndarray, ...]`: 列表或元组形式，包含多个 NumPy 数组。
                                                               数组的顺序必须与模型导出时的输入顺序一致。
                - `np.ndarray`: 单个 NumPy 数组。仅适用于模型只有一个输入的情况。

        Returns:
            Sequence[np.ndarray | SparseTensor | list | dict]:
                推理结果的序列。每个元素可以是 NumPy 数组、稀疏张量、列表或字典，
                其顺序与模型导出时的输出顺序对应。

        Raises:
            ValueError:
                - 如果 `input_data` 是列表或元组，但其长度与模型期望的输入数量不匹配。
                - 如果 `input_data` 是单个 NumPy 数组，但模型有多个输入。

        """
        feed_dict = {}

        # 处理输入格式
        if isinstance(input_data, dict):
            feed_dict = input_data

        elif isinstance(input_data, (list, tuple)):
            if len(input_data) != len(self.input_meta):
                raise ValueError(f"输入数量不匹配: 提供 {len(input_data)}, 模型需要 {len(self.input_meta)}")
            for i, data in enumerate(input_data):
                feed_dict[self.input_meta[i]['name']] = data

        else:
            # 单个 array
            if len(self.input_meta) != 1:
                raise ValueError("模型有多个输入，请使用 List 或 Dict 传参")
            feed_dict[self.input_meta[0]['name']] = input_data

        # 执行推理
        # outputs 是一个 list，顺序对应 self.output_names
        outputs = self.session.run(self.output_names, feed_dict)

        return outputs

    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, 'session'):
                del self.session
        except Exception:
            pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if hasattr(self, 'session'):
                del self.session
        except Exception:
            pass
