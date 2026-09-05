"""
从指定图像生成天空遮罩并保存。
生成的掩膜可被 apply_mask.py 复用
"""
import cv2
import numpy as np
import os

# ==================== 配置参数 ====================
REF_IMG = "/root/autodl-tmp/hdr/20251101/ASC200-2090_20251101081030_hdr.jpg"
MASK_PATH = "/root/autodl-tmp/sky_mask.png"


# ==================== 遮罩生成 ====================
def generate_mask(img_path: str) -> np.ndarray:
    """
    从单张图像生成天空遮罩

    输入:
        img_path: 输入图像的路径

    输出:
        二值遮罩 (numpy.ndarray)，255=天空，0=非天空
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 转为灰度图
    blur = cv2.GaussianBlur(gray, (7, 7), 2) # 高斯模糊去噪

    grad_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3) # 计算x梯度
    grad_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3) # 计算y梯度
    grad = np.sqrt(grad_x ** 2 + grad_y ** 2) # 计算梯度幅值
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) # 归一化到0-255

    _, grad_mask = cv2.threshold(grad, 15, 255, cv2.THRESH_BINARY_INV) # 梯度小于15的区域可能是天空
    _, bright_mask = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY) # 亮度大于80的区域可能是天空
    sky_candidate = cv2.bitwise_and(grad_mask, bright_mask) # 同时满足梯度小和亮度大的区域作为天空候选

    # 创建椭圆形结构元素，大小为7x7，用于后续形态学操作
    # 椭圆形结构元素更适合模拟自然形状，有助于更好地处理天空区域的边界
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # 执行闭运算（先膨胀后腐蚀），使用椭圆核填充天空候选区域中的小孔洞，并连接相邻的天空区域
    sky_mask = cv2.morphologyEx(sky_candidate, cv2.MORPH_CLOSE, kernel)

    # 执行开运算（先腐蚀后膨胀），使用椭圆核去除天空遮罩中的小噪点，并平滑天空区域的边界
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sky_mask, connectivity=8) # 8连通域分析，用于提取天空区域连通组件
    if num_labels <= 1:
        raise RuntimeError("未检测到任何连通域，请尝试调整阈值或更换参考图像")

    max_label = 1 + np.argmax(stats[1:, -1]) # 找到面积最大的连通域标签，假设它是天空区域
    sky_mask_final = np.zeros_like(sky_mask) # 创建一个全零的掩膜图像
    sky_mask_final[labels == max_label] = 255 # 将最大连通域区域设置为255（天空），其他区域保持为0（非天空）

    # 填充内部孔洞
    inv_mask = cv2.bitwise_not(sky_mask_final) # 取反得到非天空区域
    num_labels_inv, labels_inv, stats_inv, _ = cv2.connectedComponentsWithStats( 
        inv_mask, connectivity=8
    ) # 对非天空区域进行连通域分析，找到所有非天空区域的连通组件
    h, w = sky_mask_final.shape # 获取掩膜图像的高度和宽度
    # 遍历所有非天空连通域
    for lb in range(1, num_labels_inv): 
        # 获取当前连通域的边界框：左上角坐标 (x, y)，宽度 bw，高度 bh
        x, y, bw, bh, _ = stats_inv[lb] 
        # 检查该连通域是否完全位于图像内部（不触及边界），即为天空区域内的孔洞
        if x > 0 and y > 0 and x + bw < w and y + bh < h: 
            # 将该孔洞填充为天空（255），确保天空区域连续无空洞
            sky_mask_final[labels_inv == lb] = 255 

    return sky_mask_final


# ==================== 主程序 ====================
def main() -> None:
    print(f"参考图像: {REF_IMG}")
    mask = generate_mask(REF_IMG)
    os.makedirs(os.path.dirname(MASK_PATH), exist_ok=True)
    cv2.imwrite(MASK_PATH, mask)
    print(f"遮罩已保存至: {MASK_PATH}")
    print(f"尺寸: {mask.shape[1]}x{mask.shape[0]}, 天空像素占比: {np.sum(mask == 255) / mask.size * 100:.1f}%")


if __name__ == "__main__":
    main()