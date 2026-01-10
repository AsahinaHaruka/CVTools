"""
@Project : CVTools 
@File : base_inference.py
@Author : Haruka
@Date : 2026/1/8 16:36 
"""
import math
import os
from typing import Sequence

import onnxruntime as ort
import numpy as np
from numpy import ndarray
from onnxruntime import SparseTensor

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
    def __init__(self,
                 model_path: str,
                 enable_trt_profile: bool = False,
                 stride: int = 32,
                 max_batch_size: int = 5,
                 opt_batch_size: int = 5,
                 input_image_size: tuple[int, int] = None,
                 target_long_side: int = 640,
                 other_size: tuple[int, int, int] = (0, 100, 200)):
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
        :param other_size: 对于非图像维度的预期的输入尺寸 (min,opt,max) (仅在 enable_trt_profile=True 且模型输入动态时生效)
        """
        self.model_path = model_path
        self.stride = stride
        self.max_batch_size = max_batch_size
        self.opt_batch_size = opt_batch_size
        self.target_long_side = target_long_side

        self.fix_image=False

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        # 预分析模型结构 有几个输入，哪些是图像，哪些是动态的
        self.input_meta, self.output_names = self._scan_model_io()

        fixed_h, fixed_w = self._check_static_shape()

        if fixed_h is not None and fixed_w is not None:
            #  如果模型尺寸固定，强制使用模型尺寸
            self.img_size = (fixed_h, fixed_w)
            self.fix_image=True

            print(f"ℹ️ [Init] 检测到模型图像输入尺寸固定，强制使用固定图像输入: {self.img_size}")

        else:
            self.img_size = self._get_inference_size(
                self.target_long_side,
                stride,
                input_image_size
            )
            print(f"ℹ️ [Init] 尺寸计算: 动态模型，目标长边 {target_long_side} -> 计算推理尺寸 {self.img_size}")

        trt_provider_options = {
            'device_id': 0,
            'trt_max_workspace_size': 4 * 1024 * 1024 * 1024,  # 4GB
            'trt_fp16_enable': True,
        }

        if enable_trt_profile:
            trt_profile_options = self._build_trt_profile(other_size)
            self.fix_image=True

            if trt_profile_options:
                if trt_profile_options:
                    print(f"ℹ️ [Init] 设置TensorRT 动态形状优化参数:\n"
                          f"   Min: {trt_profile_options['trt_profile_min_shapes']}\n"
                          f"   Opt: {trt_profile_options['trt_profile_opt_shapes']}\n"
                          f"   Max: {trt_profile_options['trt_profile_max_shapes']}")

                trt_provider_options.update(trt_profile_options)

            model_name = os.path.splitext(os.path.basename(model_path))[0]
            shape_tag = f"{opt_batch_size}_{max_batch_size}__{self.img_size[0]}x{self.img_size[1]}__{other_size[0]}_{other_size[1]}_{other_size[2]}"
            cache_dir = os.path.join("trt_cache", f"{model_name}__{shape_tag}")
            os.makedirs(cache_dir, exist_ok=True)

            trt_provider_options.update({
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': cache_dir,
            })

        providers = [
            ('TensorrtExecutionProvider', trt_provider_options),
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]

        # 3. 创建正式的推理 Session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.log_severity_level = 3

        self.session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
        print(f"ℹ️ [Init] 模型加载成功! 运行设备: {self.session.get_providers()[0]}")

    def _scan_model_io(self):
        """扫描模型的输入输出元数据"""
        temp_sess = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])

        inputs = []
        for inp in temp_sess.get_inputs():
            inputs.append({
                'name': inp.name,
                'shape': inp.shape,
                'type': DTYPE_MAPPING.get(inp.type, np.float32)
            })

        output_names = [out.name for out in temp_sess.get_outputs()]
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

    @staticmethod
    def _get_inference_size(target_long: int, stride: int, input_wh: tuple[int, int] = None) -> tuple[int, int]:
        """
        统一计算推理尺寸，并强制执行 Stride 对齐检查。
        如果尺寸不满足 Stride 要求，会自动修正并打印 Warning。
        :return: [Height, Width]
        """
        if input_wh is not None:
            # 矩形推理 (根据原图比例计算)
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

        if input_wh is None:
            if new_h != target_long or new_w != target_long:
                print(
                    f"⚠️ [WARNING] 图片尺寸[{target_long}, {target_long}] 需要被 stride「{stride}」整除, 向上取整到[{new_h}, {new_w}]")

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
                        p_min.append(1)
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
        """
        执行推理。
        :param input_data:
            - Dict: {input_name: data} (推荐，最安全)
            - List: [data1, data2] (按模型 export 时的输入顺序)
            - Array: data (仅适用于单输入模型)
        :return:  list of results, every result is either a numpy array, a sparse tensor, a list or a dictionary. (对应 export 时的输出顺序)
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
