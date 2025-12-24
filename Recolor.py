import cv2
import numpy as np
import os


# ==== setup ====
BAG_PATH = os.path.join(os.path.dirname(__file__), 'bag_clean.png')  # clean background
TARGET_HEX = '#RRGGBB'  # recolor, put in the HEX code denote to your favorate color
OUT_WIDTH, OUT_HEIGHT = 1600, 1600  # size
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), f'{TARGET_HEX}.png')

# ==== tools ====
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    return (int(hex_color[4:6], 16), int(hex_color[2:4], 16), int(hex_color[0:2], 16))

def recolor_preserve_lightness(rgba, target_hex):
    if not target_hex:
        return rgba
    bgr = rgba[..., :3].copy()
    alpha = rgba[..., 3]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    tgt_bgr = np.uint8([[list(hex_to_bgr(target_hex))]])
    tgt_lab = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2LAB)[0, 0]
    a[:] = tgt_lab[1]
    b[:] = tgt_lab[2]
    recolored_bgr = cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2BGR)
    return np.dstack([recolored_bgr, alpha])

# ==== main ====
bag_rgba = cv2.imread(BAG_PATH, cv2.IMREAD_UNCHANGED)
if bag_rgba is None or bag_rgba.shape[2] != 4:
    raise ValueError("cannot read RGBA bag image")

# recolor
bag_rgba = recolor_preserve_lightness(bag_rgba, TARGET_HEX)

# calculate size
h, w = bag_rgba.shape[:2]
scale = min(OUT_WIDTH / w, OUT_HEIGHT / h)
new_w, new_h = int(w * scale), int(h * scale)
bag_resized = cv2.resize(bag_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

# white background
canvas = np.full((OUT_HEIGHT, OUT_WIDTH, 3), 255, dtype=np.uint8)

# center
x = (OUT_WIDTH - new_w) // 2
y = (OUT_HEIGHT - new_h) // 2
bgr = bag_resized[..., :3].astype(np.float32)
alpha = bag_resized[..., 3:4].astype(np.float32) / 255.0
roi = canvas[y:y+new_h, x:x+new_w].astype(np.float32)
canvas[y:y+new_h, x:x+new_w] = bgr * alpha + roi * (1 - alpha)
canvas = canvas.astype(np.uint8)

# save
cv2.imwrite(OUTPUT_PATH, canvas)
print(f"saved to: {OUTPUT_PATH}")
