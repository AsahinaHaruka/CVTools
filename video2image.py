# python
"""
@Project ：CVTools
@File ：video2image.py
@Author ：Haruka
@Date ：2025/8/16 16:45
"""
import os
import random
import sys
import time

import cv2
import json
import argparse
import numpy as np
import multiprocessing
from tqdm import tqdm

from perspective_transformation import PerspectiveTransformer

video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.webm', '.dav'}


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
        print(f"[WARN] 无法读取首帧: {video_path}")
        return None

    pts = select_four_points(frame0, f"Select 4 points - {video_key}")
    if pts is None:
        print(f"[INFO] 跳过视频(未选择点): {video_path}")
        return None

    # 使用 PerspectiveTransformer 来计算有序点和目标尺寸
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
                   worker_id: int = 0):
    time.sleep(random.uniform(0.3, 1))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] 无法打开视频: {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_val = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps_val) if fps_val and fps_val > 0 else 1

    # 使用 PerspectiveTransformer 进行变换
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
    print(f"{video_path},dst_size_w{transformer.dst_w},dst_size_h{transformer.dst_h}")

    image_count = 0
    for i in tqdm(range(frame_count), desc=f"Processing {os.path.basename(video_path)}", position=worker_id):
        ret, frame = cap.read()
        if not ret:
            break
        # 每秒提取一帧
        if i % fps != 0:
            continue

        if transformer:
            frame = transformer([frame])[0]

        image_count += 1
        frame_filename = os.path.join(output_dir, f"frame_{video_name}_{image_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

    cap.release()


def process_videos(video_dir: str, output_dir: str, enable_perspective: bool = False,
                   output_size: tuple[int, int] | None = None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    video_files = [f for f in os.listdir(video_dir)
                   if (os.path.splitext(f)[1].lower() in video_extensions and not f.startswith('.'))]
    video_files.sort()

    have_processed = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]

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

        pool = multiprocessing.Pool()  # 后台处理池
        try:
            worker_id = 0
            for vf in video_files:
                if os.path.splitext(vf)[0] in have_processed:
                    print(f"跳过已处理的视频：{vf}")
                    continue
                video_path = os.path.join(video_dir, vf)
                cfg = prepare_perspective_for_video(video_path, points_cache, output_size)
                if cfg is None:
                    continue
                output_subdir = os.path.join(output_dir, os.path.splitext(vf)[0])
                # 立即后台开始处理该视频
                pool.apply_async(extract_frames,
                                 args=(video_path, output_subdir, os.path.splitext(vf)[0], cfg, worker_id))
                worker_id += 1

            # 点选全部完成后再写缓存并等待后台任务收尾
            try:
                with open(cache_path, "w", encoding="utf-8") as fw:
                    json.dump(points_cache, fw, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[WARN] 写入缓存失败: {cache_path} -> {e}")

        except Exception as e:
            print(f"任务处理出错 -> {e}")

        print("所有点选任务已完成，正在等待后台视频处理...")
        pool.close()
        pool.join()
        print("所有后台视频处理已完成。")
    else:
        # 原并行流程（无交互）
        with multiprocessing.Pool() as pool:
            for i, video_file in enumerate(video_files):
                video_path = os.path.join(video_dir, video_file)
                output_subdir = os.path.join(output_dir, os.path.splitext(video_file)[0])
                pool.apply_async(extract_frames,
                                 args=(video_path, output_subdir, os.path.splitext(video_file)[0], None, i))
            pool.close()
            pool.join()


def parse_args():
    parser = argparse.ArgumentParser(description="Extract 1 FPS frames, optional perspective warp.")
    parser.add_argument("--video-dir", type=str,
                        help="包含视频的目录路径")
    parser.add_argument("--output-dir", type=str,
                        help="保存帧图像的目录路径")
    parser.add_argument("--perspective", action="store_true",
                        default=True,
                        help="开启首帧选点并对整段视频做透视变换")
    parser.add_argument("-oz", "--output-size", type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                        # default=[640, 640],
                        help="指定透视变换后的输出图像尺寸 (宽 高)，例如: --output-size 1920 1080")

    return parser.parse_args()


if __name__ == "__main__":
    # macOS/多进程与 OpenCV 组合更稳妥的启动方式
    multiprocessing.set_start_method("spawn", force=True)
    args = parse_args()
    if os.name == 'nt':  # Windows
        if any('\u4e00' <= ch <= '\u9fff' for ch in args.output_dir):
            print('⚠️警告：输出目录包含中文，处理后的图片可能无法写入！', file=sys.stderr)

    process_videos(args.video_dir, args.output_dir,
                   enable_perspective=args.perspective,
                   output_size=tuple(args.output_size) if args.output_size else None)
    print("Frame extraction completed.")
