from rembg import remove
import cv2
import numpy as np
import json

# ==== categories ID def ====
CIRCLE_ID = 1
BAG_ID = 2

# ==== read COCO tag ====
with open('instances_default.json', 'r', encoding='utf-8') as f:
    coco = json.load(f)

bag_seg = None
circle_seg = None

for ann in coco["annotations"]:
    if ann["category_id"] == BAG_ID:
        bag_seg = np.array(ann["segmentation"][0], dtype=np.float32).reshape(-1, 2).astype(np.int32)
    elif ann["category_id"] == CIRCLE_ID:
        circle_seg = np.array(ann["segmentation"][0], dtype=np.float32).reshape(-1, 2).astype(np.int32)

if bag_seg is None:
    raise ValueError("missing bag tag")
if circle_seg is None:
    print("no circle tag，skip circle removal")

# ==== rembg ====
input_path = 'original_bag.png'
temp_path = 'bag_no_bg.png'
output_path = 'bag_clean.png'

with open(input_path, 'rb') as i:
    output_image = remove(i.read())
with open(temp_path, 'wb') as o:
    o.write(output_image)

# ==== read RGBA ====
img = cv2.imread(temp_path, cv2.IMREAD_UNCHANGED)
bgr = img[..., :3]
alpha = np.ascontiguousarray(img[..., 3]).astype(np.uint8)

# ==== mask bag ====
mask_bag = np.zeros_like(alpha, dtype=np.uint8)
cv2.fillPoly(mask_bag, [bag_seg], 255)
alpha = cv2.bitwise_and(alpha, mask_bag)

# ==== remove circle seg ====
if circle_seg is not None:
    cv2.fillPoly(alpha, [circle_seg], 0)

# ==== finalization ====
final_img = cv2.merge([bgr[..., 0], bgr[..., 1], bgr[..., 2], alpha])
cv2.imwrite(output_path, final_img)

print(f"completed, the final image saved to {output_path}")
