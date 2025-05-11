import os
import cv2
import numpy as np

# Paths to your dataset
mask_dir = r"C:\Users\natha\CISC484_WheresWaldo\edited_images\subsample_masks\scene17"    # Your Waldo mask images
image_dir = r"C:\Users\natha\CISC484_WheresWaldo\edited_images\subsample_images\scene17"   # Your original images
label_dir = r"C:\Users\natha\CISC484_WheresWaldo\edited_images\subsample_masks_txt\scene17"  # Where the YOLO-format .txt files will go

os.makedirs(label_dir, exist_ok=True)

for mask_name in os.listdir(mask_dir):
    if not mask_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    mask_path = os.path.join(mask_dir, mask_name)
    img_path = os.path.join(image_dir, mask_name)

    # Load mask and original image to get dimensions
    mask = cv2.imread(mask_path, 0)  # Load as grayscale
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Threshold to get binary mask (in case it's not 0/255)
    _, binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    # Find contours in mask
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    label_path = os.path.join(label_dir, mask_name.rsplit('.', 1)[0] + '.txt')

    with open(label_path, 'w') as f:
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)

            # Convert to YOLO format (normalized)
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            norm_bw = bw / w
            norm_bh = bh / h

            # Class 0 = Waldo
            f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_bw:.6f} {norm_bh:.6f}\n")