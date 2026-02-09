"""
@Project ：CVTools
@File ：video2image.py
@Author ：Haruka
@Date ：2025/8/16 16:45
"""
import os
import time

import cv2
import json
import argparse
import numpy as np
import multiprocessing
from tqdm import tqdm

from utils.perspective_transformation import PerspectiveTransformer
from utils.logger import LoggerBuilder

logger = LoggerBuilder().get_logger(name="data_split")

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.webm', '.dav'}

# 容错阈值：允许连续由多少帧读取失败（监控视频坏帧常见，建议设大一点，比如100）
MAX_TOLERANCE = 100


def pool_init(lock):
    """
    进程池初始化函数：
    在每个子进程启动时运行，将主进程的锁注册给 tqdm，
    确保所有进程使用同一个锁来管理控制台输出。
    """
    tqdm.set_lock(lock)


def select_four_points(image: np.ndarray, title: str) -> np.ndarray | None:
    # 左键添加点；`r` 重置；`Enter` 确认；`Esc/q` 取消
    points: list[tuple[int, int]] = []

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, mouse_cb)

    while True:
        vis = image.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(vis, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(vis, str(i + 1), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis, "LeftClick:add | r:reset | Enter:confirm | Esc/q:cancel",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 200, 255), 2)
        cv2.imshow(title, vis)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord('q')):  # Esc/q
            cv2.destroyWindow(title)
            return None
        if key == ord('r'):
            points.clear()
        if key in (13, 10) and len(points) == 4:  # Enter
            cv2.destroyWindow(title)
            return np.array(points, dtype=np.float32)


def get_first_frame(video_path: str) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def prepare_perspective_for_video(video_path: str, cache: dict,
                                  output_size: tuple[int, int] | None = None) -> dict | None:
    video_key = os.path.basename(video_path)
    if video_key in cache:
        return cache[video_key]

    frame0 = get_first_frame(video_path)
    if frame0 is None:
        # 使用 tqdm.write 避免打断进度条
        tqdm.write(f"⚠️ 无法读取首帧: {video_path}")
        return None

    pts = select_four_points(frame0, f"Select 4 points - {video_key}")
    if pts is None:
        tqdm.write(f"⚠️ 跳过视频(未选择点): {video_path}")
        return None

    transformer = PerspectiveTransformer(points=pts, dst_size=output_size)
    rect = transformer.src_rect
    W, H = transformer.dst_w, transformer.dst_h
    data = {
        "src_points": rect.tolist(),
        "width": W,
        "height": H
    }
    cache[video_key] = data
    return data


def extract_frames(video_path: str, output_dir: str, video_name: str, persp_cfg: dict | None = None,
                   worker_id: int = 0, extract_fps: float = 1.0):
    time.sleep(worker_id * 0.1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        tqdm.write(f"❌ 无法打开: {video_name}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_val = cap.get(cv2.CAP_PROP_FPS)
    video_fps = fps_val if fps_val and fps_val > 0 else 1.0

    # 防止除以 0 或负数
    if extract_fps <= 0:
        extract_fps = 1.0

    interval = int(round(video_fps / extract_fps))
    if interval < 1:
        interval = 1

    transformer = None
    if persp_cfg:
        src_points = np.array(persp_cfg["src_points"], dtype=np.float32)
        dst_size = (persp_cfg["width"], persp_cfg["height"])
        transformer = PerspectiveTransformer(
            points=src_points,
            dst_size=dst_size,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REPLICATE
        )

    short_name = os.path.basename(video_path)
    if len(short_name) > 15:
        display_name = f"{short_name[:3]}..{short_name[-10:]}"
    else:
        display_name = short_name

    desc_str = f"W-{worker_id} {display_name}"
    image_count = 0

    consecutive_errors = 0  # 当前连续错误计数

    for i in tqdm(range(frame_count), desc=desc_str, position=worker_id, leave=True, mininterval=0.5):
        ret, frame = cap.read()

        if not ret:
            consecutive_errors += 1
            if consecutive_errors > MAX_TOLERANCE:
                # 连续坏太多帧，判定为视频真正结束
                tqdm.write(f"❌ {video_name} 结束于帧 {i} (连续错误)")
                break

            # 只是偶尔坏帧，跳过，不保存图片，继续循环找下一帧
            continue

        # 如果成功读到帧，重置错误计数器
        consecutive_errors = 0
        if i % interval != 0:
            continue

        if transformer:
            frame = transformer([frame])[0]

        image_count += 1
        frame_filename = os.path.join(output_dir, f"frame_{video_name}_{image_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

    cap.release()


def process_videos(video_dir: str, output_dir: str, enable_perspective: bool = False,
                   output_size: tuple[int, int] | None = None, extract_fps: float = 1.0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    video_files = [f for f in os.listdir(video_dir)
                   if (os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS and not f.startswith('.'))]
    video_files.sort()

    have_processed = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]

    tqdm_lock = multiprocessing.RLock()

    if enable_perspective:
        # 主进程交互取点；每得到一段视频的参数，立刻把处理任务丢到后台进程池
        cache_path = os.path.join(output_dir, "points_cache.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as fr:
                    points_cache = json.load(fr)
            except Exception:
                points_cache = {}
        else:
            points_cache = {}
        pool = multiprocessing.Pool(
            initializer=pool_init,
            initargs=(tqdm_lock,)
        )

        try:
            worker_id = 0
            for vf in video_files:
                if os.path.splitext(vf)[0] in have_processed:
                    logger.info(f"跳过已处理：{vf}")
                    continue
                video_path = os.path.join(video_dir, vf)
                cfg = prepare_perspective_for_video(video_path, points_cache, output_size)
                if cfg is None:
                    continue
                output_subdir = os.path.join(output_dir, os.path.splitext(vf)[0])

                pool.apply_async(extract_frames,
                                 args=(video_path, output_subdir, os.path.splitext(vf)[0], cfg, worker_id, extract_fps))
                worker_id += 1

            # 点选全部完成后再写缓存并等待后台任务收尾
            try:
                with open(cache_path, "w", encoding="utf-8") as fw:
                    json.dump(points_cache, fw, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"⚠️ 写入缓存失败: {e}")

        except Exception as e:
            logger.error(f"❌ 任务处理出错 -> {e}")

        logger.info("\n>>>🚀 所有点选完成，后台处理中... (请勿关闭窗口)\n")
        pool.close()
        pool.join()
        logger.info("\n所有处理已完成。")

    else:
        pool = multiprocessing.Pool(
            initializer=pool_init,
            initargs=(tqdm_lock,)
        )

        for i, video_file in enumerate(video_files):
            video_path = os.path.join(video_dir, video_file)
            output_subdir = os.path.join(output_dir, os.path.splitext(video_file)[0])
            pool.apply_async(extract_frames,
                             args=(video_path, output_subdir, os.path.splitext(video_file)[0], None, i, extract_fps))

        pool.close()
        pool.join()


def parse_args():
    parser = argparse.ArgumentParser(description="Extract 1 FPS frames, optional perspective warp.")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="包含视频的目录路径")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="保存帧图像的目录路径")
    parser.add_argument("--perspective", action="store_true",
                        default=False,
                        help="开启首帧选点并对整段视频做透视变换")
    parser.add_argument("-oz", "--output-size", type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                        help="指定透视变换后的输出图像尺寸 (宽 高)，例如: --output-size 1920 1080")
    parser.add_argument("-f", "--fps", type=float, default=1.0,
                        help="每秒提取的帧数 (默认: 1.0)")

    return parser.parse_args()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    args = parse_args()

    if args.output_size:
        args.perspective = True

    if os.name == 'nt':
        if any('\u4e00' <= ch <= '\u9fff' for ch in args.output):
            logger.warning('⚠️ 警告：输出目录包含中文，建议使用英文路径\n')

    process_videos(args.input, args.output,
                   enable_perspective=args.perspective,
                   output_size=tuple(args.output_size) if args.output_size else None,
                   extract_fps=args.fps)

    logger.info("Frame extraction completed.")
