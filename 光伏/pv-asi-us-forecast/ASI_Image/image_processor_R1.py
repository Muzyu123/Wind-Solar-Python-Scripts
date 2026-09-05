"""
使用预生成的天空遮罩批量处理图像。
需先运行 generate_mask.py 生成遮罩文件
"""
import cv2
import numpy as np
import os
from tqdm import tqdm

# ==================== 配置参数（按需修改） ====================
SRC_ROOT = "/root/autodl-tmp/hdr"
DST_ROOT = "/root/autodl-tmp/enhanced"
MASK_PATH = "/root/autodl-tmp/sky_mask.png"

# 指定日期范围（格式 YYYYMMDD），None 表示不限
DATE_START = "20251031"
DATE_END = "20251118"

# 指定每日时间段（格式 HHMM），None 表示不限
TIME_START = "0400"
TIME_END = "2000"

# CLAHE 参数
CLAHE_CLIP = 3.0 # 对比度限制系数，防止局部增强过度导致噪点放大
CLAHE_TILE = 8 # 图像划分块大小，控制局部对比度增强的粒度


# ==================== 遮罩加载 ====================
def load_mask(mask_path: str) -> np.ndarray:
    """
    加载天空遮罩文件

    输入:
        mask_path: 遮罩文件路径

    输出:
        灰度遮罩 (numpy.ndarray)，255=天空，0=非天空
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # 以灰度模式加载遮罩图像
    if mask is None:
        raise FileNotFoundError(f"无法读取遮罩: {mask_path}")
    return mask

# ==================== 图像加载 ====================
def image_loading(img_path: str) -> np.ndarray | None:
    """
    加载图像文件

    输入:
        img_path: 图像文件路径

    输出:
        图像路径列表
    """
    image_paths: list[tuple[str, str, str]] = [] # 存储符合条件的图像路径列表，元素为 (日期目录, 文件名, 完整路径)
    date_dirs = sorted(os.listdir(img_path)) # 获取源目录下的日期子目录列表，并排序
    for date_dir in date_dirs:
        src_date_path = os.path.join(img_path, date_dir)
        if not os.path.isdir(src_date_path):
            continue
        for fname in sorted(os.listdir(src_date_path)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            ts = extract_timestamp(fname)
            if not in_time_window(ts, DATE_START, DATE_END, TIME_START, TIME_END):
                continue
            image_paths.append((date_dir, fname, os.path.join(src_date_path, fname)))
    
    return image_paths

# ==================== 图像处理 ====================
def process_image(img_path: str, mask: np.ndarray) -> np.ndarray | None:
    """
    使用预生成遮罩处理单张图像，应用 CLAHE 增强并合成 BGRA 通道

    输入:
        img_path: 输入图像的路径
        mask: 天空遮罩

    输出:
        enhanced_bgra (numpy.ndarray) 或 None（读取失败时）
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转为灰度图
    # 创建 CLAHE 对象，增强图像对比度
    # clipLimit: 对比度限制系数，防止局部增强过度导致噪点放大
    # tileGridSize: 将图像划分为多个小块，分别进行直方图均衡，控制局部对比度增强的粒度
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE))
    enhanced_gray = clahe.apply(gray) # 对灰度图像应用 CLAHE 增强
    enhanced_bgra = cv2.merge([enhanced_gray, enhanced_gray, enhanced_gray, mask]) # 将增强后的灰度图复制到 BGR 三个通道，并将遮罩作为 Alpha 通道合成 BGRA 图像

    return enhanced_bgra


# ==================== 时间戳工具 ====================
def extract_timestamp(fname: str) -> str | None:
    """
    从文件名中提取时间戳

    输入:
        fname: 文件名（格式如 ASC200-XXXX_YYYYMMDDHHMMSS_hdr.jpg）

    输出:
        时间戳字符串（YYYYMMDDHHMMSS），解析失败返回 None
    """
    try:
        base = os.path.splitext(fname)[0]
        parts = base.split("_")
        return parts[1]
    except (IndexError, ValueError):
        return None


def in_time_window(ts_str: str | None, date_start: str | None, date_end: str | None,
                   time_start: str | None, time_end: str | None) -> bool:
    """
    判断时间戳是否在指定的日期和时间窗口内

    输入:
        ts_str: 时间戳字符串（YYYYMMDDHHMMSS）
        date_start: 起始日期（YYYYMMDD），None 不限
        date_end: 结束日期（YYYYMMDD），None 不限
        time_start: 起始时间（HHMM），None 不限
        time_end: 结束时间（HHMM），None 不限

    输出:
        是否在窗口内 (bool)
    """
    if ts_str is None:
        return False
    date_part = ts_str[:8] 
    time_part = ts_str[8:12] 
    if date_start and date_part < date_start:
        return False
    if date_end and date_part > date_end:
        return False
    if time_start and time_part < time_start:
        return False
    if time_end and time_part > time_end:
        return False
    return True


# ==================== 主程序 ====================
def main() -> None:
    """
    主入口：加载遮罩，扫描并过滤图像，批量处理并保存
    """
    mask = load_mask(MASK_PATH)
    print(f"已加载遮罩: {MASK_PATH}")

    image_paths = image_loading(SRC_ROOT)

    if not image_paths:
        print("未找到符合条件的图像，请检查日期/时间参数。")
        return

    print(f"共找到 {len(image_paths)} 张图像待处理")

    fail_list: list[str] = []
    for date_dir, fname, src_path in tqdm(image_paths, desc="处理进度"):
        enhanced = process_image(src_path, mask)
        if enhanced is None:
            fail_list.append(os.path.join(date_dir, fname)) 
            continue

        dst_dir = os.path.join(DST_ROOT, date_dir)
        os.makedirs(dst_dir, exist_ok=True)
        dst_name = os.path.splitext(fname)[0] + ".png" # 输出为 PNG 格式以保留 Alpha 通道
        cv2.imwrite(os.path.join(dst_dir, dst_name), enhanced)

    print(f"处理完成！成功: {len(image_paths) - len(fail_list)}, 失败: {len(fail_list)}")
    if fail_list:
        print("失败文件:")
        for f in fail_list:
            print(f"  {f}")


if __name__ == "__main__":
    main()
