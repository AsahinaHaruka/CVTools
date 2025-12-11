"""
@Project ：CVTools
@File ：color_to_gray.py
@Author ：Haruka
@Date ：2025/9/26 11:38 
"""
import argparse
import sys
from pathlib import Path
from typing import Iterable, List
from tqdm import tqdm

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="批量将彩色图片转为灰度（OpenCV）")
    p.add_argument("--src", help="输入目录路径")
    p.add_argument("--dst", help="输出目录路径")
    p.add_argument("--recursive", action="store_true", help="递归处理子目录")
    p.add_argument("--overwrite", action="store_true", help="若输出已存在则覆盖")
    return p.parse_args()


def is_subpath(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def list_images(src: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from (p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    else:
        yield from (p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def to_gray(in_path: Path, out_path: Path, overwrite: bool) -> bool:
    if out_path.exists() and not overwrite:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        img = cv2.imread(str(in_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("无法读取图像")

        # 转灰度（自动适配 BGR/BGRA/灰度）
        if img.ndim == 2:
            gray = img
        elif img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        params = []
        ext = out_path.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        elif ext == ".png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        elif ext == ".webp":
            params = [cv2.IMWRITE_WEBP_QUALITY, 95]

        # 若保存为 BMP，确保为 8-bit 灰度
        if ext in {".bmp", ".dib"} and gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            gray = gray.astype(np.uint8)

        ok = cv2.imwrite(str(out_path), gray, params)
        if not ok:
            raise ValueError("保存失败")
        return True
    except Exception as e:
        print(f"[错误] 转换失败: {in_path} -> {out_path} ({e})", file=sys.stderr)
        return False


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists() or not src.is_dir():
        print(f"[错误] 输入目录不存在或不可用: {src}", file=sys.stderr)
        sys.exit(1)

    if is_subpath(dst, src):
        print(f"[错误] 输出目录 `{dst}` 不可位于输入目录 `{src}` 内，请选择不同位置。", file=sys.stderr)
        sys.exit(1)

    dst.mkdir(parents=True, exist_ok=True)

    files: List[Path] = list(list_images(src, args.recursive))
    if not files:
        print("[提示] 未找到待处理的图片文件。")
        return

    converted = 0
    skipped = 0
    for f in tqdm(files):
        rel = f.relative_to(src) if args.recursive else f.name
        out_path = dst / rel
        if to_gray(f, out_path, args.overwrite):
            converted += 1
        else:
            skipped += 1

    print(f"[完成] 共发现 {len(files)} 张图片，转换 {converted}，跳过 {skipped}。输出目录: {dst}")


if __name__ == "__main__":
    main()