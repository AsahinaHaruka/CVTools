"""
@Project : CVTools 
@File : base_inference.py
@Author : Haruka
@Date : 2026/1/8 16:36 
"""
import math
import os

import onnxruntime as ort
import numpy as np

# 使用字典映射动态获取模型输入的数据类型
DTYPE_MAPPING = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(int8)": np.int8
}


class ONNXInference:
    def __init__(self,
                 model_path: str,
                 enable_trt_profile: bool = False,
                 max_batch_size: int = 5,
                 opt_batch_size: int = 5,
                 input_image_size: tuple[int, int] = None,
                 target_long_side: int = 640):
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
        self.model_path = model_path
        self.stride = 32
        self.max_batch_size = max_batch_size
        self.opt_batch_size = opt_batch_size
        self.target_long_side = target_long_side

        # 这里的 img_size 作为正方形托底
        self.img_size = [target_long_side, target_long_side]

        if enable_trt_profile:
            self.is_fixed_size = True
            if input_image_size is not None:
                self.img_size = self._cal_optimal_fixed_shape(input_image_size, target_long_side, self.stride)
            print(f"ℹ️ [Init] 模式: TensorRT固定形状优化。锁定引擎尺寸: {self.img_size}")

        else:
            self.is_fixed_size = False
            print(f"ℹ️ [Init] 模式: 动态形状 (Standard Dynamic Rect)。目标长边: {target_long_side}")

        trt_profile_options = self._analyze_model_and_get_profile(enable_trt_profile)

        trt_provider_options = {
            'device_id': 0,
            'trt_max_workspace_size': 4294967296,  # 4GB
            'trt_fp16_enable': True,
        }

        if enable_trt_profile:
            try:
                model_name = os.path.splitext(os.path.basename(model_path))[0]
                shape_tag = f"{self.img_size[0]}x{self.img_size[1]}" if self.is_fixed_size else "dynamic"
                cache_dir = f"./trt_cache/{model_name}_{shape_tag}"
                os.makedirs(cache_dir, exist_ok=True)
            except:
                cache_dir = "."

            cache_profile = {
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': cache_dir,
            }

            trt_provider_options.update(cache_profile)

            # 如果启用了 Profile 优化，将生成的 shape 配置合并进去
            if trt_profile_options:
                print(f"ℹ️ [Init] 设置TensorRT 动态形状优化参数:\n"
                      f"   Min: {trt_profile_options['trt_profile_min_shapes']}\n"
                      f"   Opt: {trt_profile_options['trt_profile_opt_shapes']}\n"
                      f"   Max: {trt_profile_options['trt_profile_max_shapes']}")
                trt_provider_options.update(trt_profile_options)

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

    @staticmethod
    def _cal_optimal_fixed_shape(input_wh: tuple, target_long: int, stride: int) -> list:
        """
        计算最佳的矩形推理尺寸。
        :return: [Height, Width]
        """
        w, h = input_wh

        # 矩形推理优化逻辑
        scale = target_long / max(w, h)

        # 缩放并向上取整到 stride 的倍数
        new_h = int(math.ceil(h * scale / stride) * stride)
        new_w = int(math.ceil(w * scale / stride) * stride)

        return [new_h, new_w]

    def _analyze_model_and_get_profile(self, enable_profile: bool) -> dict:
        """获取模型输入名、动态Shape信息，并生成 Profile 字符串"""
        temp_sess = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        input_info = temp_sess.get_inputs()[0]
        output_info = temp_sess.get_outputs()[0]

        self.input_name = input_info.name
        self.output_name = output_info.name
        self.dtype = DTYPE_MAPPING.get(input_info.type, np.float32)
        model_shape = input_info.shape
        del temp_sess

        if isinstance(model_shape[2], int) and isinstance(model_shape[3], int):
            self.img_size = [model_shape[2], model_shape[3]]
            self.is_fixed_size = True  # 模型本身固定，强制变为固定模式
            print(f"⚠️ 检测到模型为静态尺寸 {self.img_size}，已强制切换为固定尺寸模式。")

        if not enable_profile: return {}

        min_dims, opt_dims, max_dims = [], [], []

        # 遍历维度生成配置
        for idx, dim in enumerate(model_shape):
            if isinstance(dim, int):  # 静态维度
                min_dims.append(dim)
                opt_dims.append(dim)
                max_dims.append(dim)
            elif isinstance(dim, str):  # 动态维度
                if idx == 0:  # Batch
                    min_dims.append(1)
                    opt_dims.append(self.max_batch_size)
                    max_dims.append(self.opt_batch_size)
                elif idx == 2:  # Height
                    min_dims.append(self.img_size[0])
                    opt_dims.append(self.img_size[0])
                    max_dims.append(self.img_size[0])
                elif idx == 3:  # Width
                    min_dims.append(self.img_size[1])
                    opt_dims.append(self.img_size[1])
                    max_dims.append(self.img_size[1])
                else:
                    min_dims.append(1)
                    opt_dims.append(1)
                    max_dims.append(1)

        def to_str(dims):
            return f"{self.input_name}:{'x'.join(map(str, dims))}"

        return {
            'trt_profile_min_shapes': to_str(min_dims),
            'trt_profile_opt_shapes': to_str(opt_dims),
            'trt_profile_max_shapes': to_str(max_dims)
        }

    def __call__(self, input_data: list[np.ndarray] | np.ndarray) -> np.ndarray:
        """
        Perform inference on the input data.
        :param input_data: Input data for inference. input_data :[batch_size, height, width, channels]
        :return: Inference results
        """

        # 执行推理
        outputs = self.session.run([self.output_name], {self.input_name: input_data})

        return outputs[0].astype(np.float32)
