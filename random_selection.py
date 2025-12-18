"""
@Project ：CVTools
@File ：random_selection.py
@Author ：Haruka
@Date ：2025/10/7 08:29 
"""
import argparse
import random
import shutil
from pathlib import Path

image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}


def _get_image_count(src_dir: Path) -> int:
    """计算文件夹内图片数量"""
    return len([f for f in src_dir.iterdir() if
                (f.is_file() and f.suffix.lower() in image_extensions and not f.name.startswith('.'))])


def _select_random_files(src_dir: Path, dst_dir: Path, num_files: int):
    """
    具体执行随机选择和复制的函数
    """
    # 获取所有文件的列表
    files = [f for f in src_dir.iterdir() if
             (f.is_file() and f.suffix.lower() in image_extensions and not f.name.startswith('.'))]

    if len(files) == 0:
        return

    # 如果需要抽取的数量为0，直接跳过
    if num_files <= 0:
        return

    if len(files) < num_files:
        print(f"⚠️警告：源文件夹 '{src_dir}' 中只有 {len(files)} 个文件，不足 {num_files} 个,将全部复制。")
        num_files = len(files)

    # 随机选择文件
    selected_files = random.sample(files, num_files)

    # 复制文件
    print(f"正在从 '{src_dir}' 向 '{dst_dir}' 复制 {num_files} 个随机文件...")
    for file_path in selected_files:
        # 直接复制到目标文件夹，同名文件将直接覆盖
        shutil.copy(file_path, dst_dir)


def sample_files_from_directories(src_dir: Path, dst_dir: Path, num_files: int, mode: str):
    """
    主逻辑控制函数
    :param mode: 'fixed' (每个文件夹n张) 或 'proportional' (总共n张，按比例分配)
    """
    if not src_dir.is_dir():
        print(f"错误：源文件夹 '{src_dir}' 不存在。")
        return

    # 创建目标文件夹
    dst_dir.mkdir(exist_ok=True)

    # 收集所有需要处理的文件夹（子文件夹 + 根目录）
    target_dirs = [d for d in src_dir.iterdir() if (d.is_dir() and d != dst_dir)]
    target_dirs.append(src_dir)

    print(f"正在处理 {len(target_dirs)} 个文件夹路径...")

    # ==========================
    # 模式 A: 按比例抽取总量
    # ==========================
    if mode == 'proportional':
        print(f"🔵 模式[proportional]：按比例抽取，总目标数量：{num_files}")

        # 1. 统计每个文件夹的图片数量
        dir_counts = {}
        total_images = 0
        for d in target_dirs:
            count = _get_image_count(d)
            if count > 0:
                dir_counts[d] = count
                total_images += count

        if total_images == 0:
            print("❌ 所有文件夹中均未发现图片。")
            return

        print(f"📊 发现总图片数: {total_images}。")

        # 如果请求的总数大于现有总数，全部复制
        if num_files >= total_images:
            print("⚠️ 请求数量大于或等于总图片数，将复制所有图片。")
            for d, count in dir_counts.items():
                _select_random_files(d, dst_dir, count)
            return

        # 2. 计算每个文件夹应该分配的数量
        allocations = {}
        current_allocated_sum = 0

        # 初步分配 (向下取整)
        for d, count in dir_counts.items():
            ratio = count / total_images
            allocate = int(ratio * num_files)
            allocations[d] = allocate
            current_allocated_sum += allocate

        # 3. 处理剩余的配额 (填补因向下取整导致的缺口)
        remainder = num_files - current_allocated_sum
        if remainder > 0:
            # 找出还有剩余图片没被抽完的文件夹
            available_dirs = [d for d in dir_counts.keys() if allocations[d] < dir_counts[d]]
            while remainder > 0 and available_dirs:
                lucky_dir = random.choice(available_dirs)
                if allocations[lucky_dir] < dir_counts[lucky_dir]:
                    allocations[lucky_dir] += 1
                    remainder -= 1
                else:
                    available_dirs.remove(lucky_dir)

        # 4. 执行复制
        for d, alloc_num in allocations.items():
            if alloc_num > 0:
                print(f"-> 文件夹 '{d.name}': 总数 {dir_counts[d]}, 抽取 {alloc_num}")
                _select_random_files(d, dst_dir, alloc_num)

    # ==========================
    # 模式 B: 每个文件夹固定数量 (默认)
    # ==========================
    elif mode == 'fixed':
        print(f"🔵 模式[fixed]：每个文件夹固定抽取 {num_files} 张")
        for subdir in target_dirs:
            if _get_image_count(subdir) > 0:
                _select_random_files(subdir, dst_dir, num_files)

    # ==========================
    # 扩展模式接口...
    # ==========================
    else:
        print(f"❌ 未知模式: {mode}")

    print("✅ 所有任务完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="随机抽取图片工具。")
    parser.add_argument('-i', '--input', type=str, required=True, help="源目录路径。")
    parser.add_argument('-o', '--output', type=str, required=True, help="目标目录路径。")
    parser.add_argument('-n', '--num', type=int, required=True, help="抽取数量。")

    # 修改：使用 --model 指定模式，默认为 fixed
    parser.add_argument('--model', type=str, default='fixed', choices=['fixed', 'proportional'],
                        help="抽取模式: 'fixed' (默认, 每个文件夹抽n张) 或 'proportional' (总共抽n张，按比例分配)。")

    args = parser.parse_args()

    sample_files_from_directories(
        Path(args.input),
        Path(args.output),
        args.num,
        args.model
    )
