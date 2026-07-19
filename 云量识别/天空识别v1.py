import cv2
import numpy as np
import matplotlib.pyplot as plt

# ====================== 1. 读取图像（直接转灰度，抛弃颜色） ======================
img_path = "/work/zfshu/24learn/hyx-test/大创/全天空成像仪/天空识别/resource/Image_0001782285634614.jpg"
img = cv2.imread(img_path)
# 直接转灰度 = 完全不依赖颜色！
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ====================== 2. 无颜色天空分割核心 ======================
# 1) 高斯模糊：去除噪点，强化天空平滑特性
blur = cv2.GaussianBlur(gray, (7, 7), 2)

# 2) 梯度计算：天空梯度=0，树木/建筑梯度很大
grad_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
grad = np.sqrt(grad_x**2 + grad_y**2)
grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# 3) 梯度阈值：梯度小 = 天空（平滑），梯度大 = 遮挡物
# 阈值越低，识别天空越宽松；越高越严格
_, grad_mask = cv2.threshold(grad, 15, 255, cv2.THRESH_BINARY_INV)

# 4) 亮度阈值：天空通常更亮（双保险）
_, bright_mask = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY)

# 5) 梯度+亮度 结合 = 天空候选区
sky_candidate = cv2.bitwise_and(grad_mask, bright_mask)

# 6) 形态学优化：去噪点、补空洞
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
sky_mask = cv2.morphologyEx(sky_candidate, cv2.MORPH_CLOSE, kernel)
sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)

# 7) 保留最大连通域：天空一定是画面最大的平滑区域（最关键！）
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sky_mask, connectivity=8)
max_label = 1 + np.argmax(stats[1:, -1])  # 找面积最大区域
sky_mask_final = np.zeros_like(sky_mask)
sky_mask_final[labels == max_label] = 255

# 8) 外轮廓填充：消除天空内部空洞（云纹理误判的洞），得到完整连续天空
contours, _ = cv2.findContours(sky_mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sky_mask_solid = np.zeros_like(sky_mask_final)
cv2.drawContours(sky_mask_solid, contours, -1, 255, -1)  # -1厚度=填满
sky_mask_final = sky_mask_solid

# ====================== 3. 提取天空 + 增强云边缘 ======================
# 提取纯天空
sky_only = cv2.bitwise_and(img, img, mask=sky_mask_final)

# 灰度化 + CLAHE增强（强化云纹理）
gray_sky = cv2.cvtColor(sky_only, cv2.COLOR_BGR2GRAY)
gray_sky[sky_mask_final == 0] = 0
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
enhanced = clahe.apply(gray_sky)

# ====================== 4. 显示结果 ======================
plt.figure(figsize=(16,10))
plt.subplot(231), plt.imshow(img_rgb), plt.title('Original Image'), plt.axis('off') # 原始图像
plt.subplot(232), plt.imshow(grad, cmap='gray'), plt.title('Gradient Map (Obstacle Highlighted)'), plt.axis('off') # 梯度图（遮挡物高亮）
plt.subplot(233), plt.imshow(sky_mask_final, cmap='gray'), plt.title('Sky Mask (No Color)'), plt.axis('off') # 无颜色天空遮罩
plt.subplot(234), plt.imshow(cv2.cvtColor(sky_only, cv2.COLOR_BGR2RGB)), plt.title('Pure Sky (No Color)'), plt.axis('off') # 纯天空（无颜色）
plt.subplot(235), plt.imshow(gray_sky, cmap='gray'), plt.title('Sky Map (No Color)'), plt.axis('off') # 灰度化天空（无颜色）
plt.subplot(236), plt.imshow(enhanced, cmap='gray'), plt.title('Enhanced Cloud (No Color)'), plt.axis('off') # 增强云（无颜色）
plt.tight_layout()
plt.show()

# ====================== 5. 保存 ======================
cv2.imwrite("/work/zfshu/24learn/hyx-test/大创/全天空成像仪/天空识别/results/sky_mask_no_color.png", sky_mask_final)
cv2.imwrite("/work/zfshu/24learn/hyx-test/大创/全天空成像仪/天空识别/results/pure_sky_no_color.png", sky_only)
cv2.imwrite("/work/zfshu/24learn/hyx-test/大创/全天空成像仪/天空识别/results/enhanced_cloud_no_color.png", enhanced)
