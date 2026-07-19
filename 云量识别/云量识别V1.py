import cv2
import numpy as np
import os

# ========================= 配置区 =========================
IMG_PATH = "/work/zfshu/24learn/hyx-test/大创/全天空成像仪/天空识别/resource/Image_0001782285514474.jpg"
OUT_DIR = "/work/zfshu/24learn/hyx-test/大创/全天空成像仪/天空识别/results"

GAUSSIAN_KERNEL = (7, 7)
GAUSSIAN_SIGMA = 2
GRAD_SKY_THRESH = 15
MORPH_KERNEL_SIZE = 7

BRIGHT_CLEAR_PCT  = 65
GRAD_CLEAR_PCT    = 30
GRAD_THICK_PCT    = 70
BRIGHT_THICK_PCT  = 35
CLEAR_NONCLEAR_GAP = 40   # Clear与非Clear亮度差低于此 → 阴天无蓝天

COLOR_CLEAR  = (200, 150, 80)
COLOR_THIN   = (120, 200, 120)
COLOR_THICK  = (80, 80, 220)
COLOR_NONSKY = (40, 40, 40)


# ========================= 1. 读取 =========================
print(f"[1/5] 读取: {IMG_PATH}")
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"图像未找到: {IMG_PATH}")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape


# ========================= 2. 天空分割 =========================
print("[2/5] 天空分割...")
blur = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)
grad_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
grad = np.sqrt(grad_x**2 + grad_y**2)
grad_u8 = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

_, grad_mask = cv2.threshold(grad_u8, GRAD_SKY_THRESH, 255, cv2.THRESH_BINARY_INV)
smooth_vals = blur[grad_mask > 0]
th_bright = cv2.threshold(smooth_vals, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0] \
            if len(smooth_vals) > 0 else 80
print(f"     亮度阈值(Otsu)={th_bright:.0f}")
_, bright_mask = cv2.threshold(blur, th_bright, 255, cv2.THRESH_BINARY)
sky_candidate = cv2.bitwise_and(grad_mask, bright_mask)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
sky_mask = cv2.morphologyEx(sky_candidate, cv2.MORPH_CLOSE, kernel)
sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sky_mask, connectivity=8)
max_label = 1 + np.argmax(stats[1:, -1])
sky_mask_final = np.zeros_like(sky_mask)
sky_mask_final[labels == max_label] = 255

contours_sky, _ = cv2.findContours(sky_mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sky_solid = np.zeros_like(sky_mask_final)
cv2.drawContours(sky_solid, contours_sky, -1, 255, -1)

sky_area = np.sum(sky_solid == 255)
print(f"     天空面积: {sky_area} px ({sky_area/(h*w)*100:.1f}%)")


# ========================= 3. 云分类 =========================
print("[3/5] 云分类（无云/淡云/厚云）...")
sky_idx = sky_solid == 255
b = blur[sky_idx].astype(np.float32)
g = grad_u8[sky_idx].astype(np.float32)

th_bright_clear = np.percentile(b, BRIGHT_CLEAR_PCT)
th_grad_clear   = np.percentile(g, GRAD_CLEAR_PCT)
th_grad_thick   = np.percentile(g, GRAD_THICK_PCT)
th_bright_thick = np.percentile(b, BRIGHT_THICK_PCT)

print(f"     自适应: clear(b>={th_bright_clear:.0f}, g<={th_grad_clear:.0f})")
print(f"             thick(g>={th_grad_thick:.0f} 或 b<={th_bright_thick:.0f})")

cloud_class = np.zeros((h, w), dtype=np.uint8)
result = np.full(len(b), 2, dtype=np.uint8)
result[(b >= th_bright_clear) & (g <= th_grad_clear)] = 1
result[g >= th_grad_thick] = 3
result[b <= th_bright_thick] = 3
cloud_class[sky_idx] = result

cloud_class = cv2.medianBlur(cloud_class, 3)
cloud_class[sky_solid == 0] = 0
cloud_class[(cloud_class == 0) & sky_solid.astype(bool)] = 2

# 阴天检测：Clear区与非Clear区亮度差太小 → 全阴天，无真正蓝天
clear_mask = cloud_class == 1
if clear_mask.sum() > 0:
    clear_median = np.median(blur[clear_mask])
    nonclear_median = np.median(blur[(cloud_class == 2) | (cloud_class == 3)])
    gap = clear_median - nonclear_median
    if gap < CLEAR_NONCLEAR_GAP:
        print(f"     ⚠ Clear与非Clear亮度差={gap:.0f}<{CLEAR_NONCLEAR_GAP} → 全阴天，降为淡云")
        cloud_class[cloud_class == 1] = 2
    else:
        print(f"     亮度差={gap:.0f} ≥ {CLEAR_NONCLEAR_GAP}，保留Clear")

n_clear = (cloud_class == 1).sum()
n_thin  = (cloud_class == 2).sum()
n_thick = (cloud_class == 3).sum()
total = n_clear + n_thin + n_thick
if total > 0:
    print(f"     无云: {n_clear:>8} px ({n_clear/total*100:5.1f}%)")
    print(f"     淡云: {n_thin:>8} px ({n_thin/total*100:5.1f}%)")
    print(f"     厚云: {n_thick:>7} px ({n_thick/total*100:5.1f}%)")
    print(f"     总云量: {n_thin+n_thick:>7} px ({(n_thin+n_thick)/total*100:5.1f}%)")


# ========================= 4. 色块标记 =========================
print("[4/5] 生成色块图...")
color_img = np.full_like(img, COLOR_NONSKY)
color_img[cloud_class == 1] = COLOR_CLEAR
color_img[cloud_class == 2] = COLOR_THIN
color_img[cloud_class == 3] = COLOR_THICK
overlay = cv2.addWeighted(img, 0.6, color_img, 0.4, 0)
contour_img = img.copy()
for cid, col in [(1, COLOR_CLEAR), (2, COLOR_THIN), (3, COLOR_THICK)]:
    m = (cloud_class == cid).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_img, cnts, -1, col, 2)


# ========================= 5. 保存 =========================
print("[5/5] 保存...")

def draw_legend(img_target):
    h_img, w_img = img_target.shape[:2]
    LEGEND_AREA_RATIO = 0.05
    ASPECT = 0.43
    total_px = w_img * h_img
    box_w = int(np.sqrt(total_px * LEGEND_AREA_RATIO / ASPECT))
    box_h = int(box_w * ASPECT)
    box_w = min(box_w, w_img - 20)
    box_h = min(box_h, h_img - 20)
    margin = max(8, box_w // 15)
    x0 = w_img - box_w - margin
    y0 = margin
    font_scale = box_w / 480
    thick = max(2, int(box_w / 220))
    fs = cv2.FONT_HERSHEY_SIMPLEX

    roi = img_target[y0:y0+box_h, x0:x0+box_w].astype(np.float32)
    bg = np.full((box_h, box_w, 3), (35, 35, 35), dtype=np.float32)
    blended = roi * 0.45 + bg * 0.55
    img_target[y0:y0+box_h, x0:x0+box_w] = blended.astype(np.uint8)
    cv2.rectangle(img_target, (x0, y0), (x0+box_w, y0+box_h), (100,100,100), thick)

    title = "Cloud Type"
    (tw, th), _ = cv2.getTextSize(title, fs, font_scale*1.1, thick)
    cv2.putText(img_target, title, (x0 + (box_w - tw)//2, y0 + th + int(box_h*0.06)),
                fs, font_scale*1.1, (220,220,220), thick)

    row_h = int(box_h * 0.7 / 3)
    start_y = y0 + int(box_h * 0.28)
    swatch_w = int(box_w * 0.18)
    swatch_h = int(row_h * 0.55)
    text_x = x0 + swatch_w + int(box_w * 0.06)

    items = [
        ("Clear",  COLOR_CLEAR,  n_clear/total*100),
        ("Thin",   COLOR_THIN,   n_thin/total*100),
        ("Thick",  COLOR_THICK,  n_thick/total*100),
    ]
    for i, (label, color, pct) in enumerate(items):
        cy = start_y + i * row_h
        cv2.rectangle(img_target, (x0+int(box_w*0.1), cy),
                      (x0+int(box_w*0.1)+swatch_w, cy+swatch_h), color, -1)
        cv2.rectangle(img_target, (x0+int(box_w*0.1), cy),
                      (x0+int(box_w*0.1)+swatch_w, cy+swatch_h), (0,0,0), max(1, thick//2))
        cv2.putText(img_target, f"{label}  {pct:.1f}%",
                    (text_x, cy + swatch_h - int(swatch_h*0.2)),
                    fs, font_scale, (255,255,255), thick)
    return img_target

os.makedirs(OUT_DIR, exist_ok=True)
base = os.path.splitext(os.path.basename(IMG_PATH))[0]
fs = cv2.FONT_HERSHEY_SIMPLEX

cv2.imwrite(f"{OUT_DIR}/{base}_class.png",   draw_legend(color_img.copy()))
cv2.imwrite(f"{OUT_DIR}/{base}_overlay.png", draw_legend(overlay.copy()))
cv2.imwrite(f"{OUT_DIR}/{base}_contour.png", draw_legend(contour_img.copy()))
cv2.imwrite(f"{OUT_DIR}/{base}_mask.png", sky_solid)

scale_cmp = min(800 / max(h, w), 1.0)
cmp_w, cmp_h = int(w * scale_cmp), int(h * scale_cmp)
img_cmp = cv2.resize(img, (cmp_w, cmp_h))
ovl_cmp = cv2.resize(overlay, (cmp_w, cmp_h))
sep_line = np.ones((cmp_h, 4, 3), dtype=np.uint8) * 100
label_bar = np.ones((32, cmp_w * 2 + 4, 3), dtype=np.uint8) * 255
cv2.putText(label_bar, "Original", (cmp_w//2-40, 22), fs, 0.65, (0,0,0), 2)
cv2.putText(label_bar, "Cloud Classification", (cmp_w+cmp_w//2-105, 22), fs, 0.65, (0,0,0), 2)
ovl_cmp = draw_legend(ovl_cmp)
pair = np.hstack([img_cmp, sep_line, ovl_cmp])
compare = np.vstack([label_bar, pair])
cv2.imwrite(f"{OUT_DIR}/{base}_compare.png", compare)

print(f"\n✓ 完成 → {OUT_DIR}/")
print(f"  {base}_class.png    纯色块")
print(f"  {base}_overlay.png  半透明叠加")
print(f"  {base}_contour.png  轮廓线")
print(f"  {base}_compare.png  原图vs叠加对比")
print(f"  {base}_mask.png     天空遮罩")
