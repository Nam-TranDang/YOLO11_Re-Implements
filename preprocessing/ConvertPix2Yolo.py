import numpy as np
import logging

# Convert from x1 y1 x2 y2 --> to YOLO format
def Pix2Yolo (boxes_np, img_width, img_height, idx):
    y = np.copy(boxes_np[:, 1:]) # Lấy tất cả ngoài trừ class ở cột 0 
    logging.info(f"Sample {idx} - y (x1, y1, x2, y2) before YOLO conversion: {y[:3]}")            

    y[:, 0] = ((boxes_np[:, 1] + boxes_np[:, 3]) / 2) / img_width  # center x
    y[:, 1] = ((boxes_np[:, 2] + boxes_np[:, 4]) / 2) / img_height  # center y
    y[:, 2] = (boxes_np[:, 3] - boxes_np[:, 1]) / img_width  # width
    y[:, 3] = (boxes_np[:, 4] - boxes_np[:, 2]) / img_height  # height
    return y