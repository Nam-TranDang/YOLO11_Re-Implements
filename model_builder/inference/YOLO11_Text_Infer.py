# python3 -m .inference.YOLO11_Text_Infer

import torch
import cv2
from PIL import Image

import numpy as np
import os
from pathlib import Path

import yaml

# --- Assuming the provided model code is in a file named 'yolo_v11.py' ---

# --- Imports from your custom project structure ---
from model_builder.registry.YOLO11_Registry import Get_Yolo11_Model
from model_builder.utils.YOLO11_Img_Utils import Letterbox_resize
from model_builder.postprocessing.NonMaxSurpression import non_max_suppression
from model_builder.postprocessing.ConvertYolo2Pix import Yolo2Pix

# det[:, :4] = scale_boxes(resized_shape, det[:, :4], img0_np.shape, ratio_pad=(ratio, padding_values))

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """
    Rescales bounding boxes (xyxy) from the resized image shape to the original image shape.
    """
    ratio, pad = ratio_pad
    # gain = ratio  # FIXED: ratio is already the min ratio (float)
    pad_x, pad_y = pad

    boxes[:, [0, 2]] -= pad_x  # x padding
    boxes[:, [1, 3]] -= pad_y  # y padding
    boxes[:, :4] /= ratio
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, shape):
    """
    Clips bounding box (xyxy) coordinates to fit within the image shape (height, width).
    This is a safety measure to prevent out-of-bounds coordinates after scaling.
    """
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, shape[1])  # x1 (width)
        boxes[:, 1].clamp_(0, shape[0])  # y1 (height)
        boxes[:, 2].clamp_(0, shape[1])  # x2 (width)
        boxes[:, 3].clamp_(0, shape[0])  # y2 (height)
    else: # numpy
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])

def infer(image_path, model_weights_path, config):
    """
    Performs inference on an image using a YOLOv11 model for text detection.

    Args:
        image_path (str): Path to the input image.
        model_weights_path (str): Path to the model weights file.
        config (dict): Configuration dictionary.
    """
    # Load model from your registry
    model_name = config["model"]["variant"]
    num_classes = config["model"]["nc"]
    model = Get_Yolo11_Model(model_name, num_classes)
    
    # Load weights
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu'))['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Read img in PIL + Resize letterbox 
    img0 = Image.open(image_path).convert('RGB')
    target_size = (config['input_dim']['height'], config['input_dim']['width'])
    img, ratio, padding_values = Letterbox_resize(img0, target_img_size=target_size)
    
    # Convert PIL Image --> NumPy array for model input
    img = np.array(img)

    # Get resized shape BEFORE transpose
    resized_shape = img.shape[:2]  # (H, W) - correct!


    # Convert image for model input
    img = img.transpose((2, 0, 1))   # Current image in PIL -- not cv2 
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(device)
    # Set tensor type to float32 as requested
    img_tensor = img_tensor.to(torch.float32)
    img_tensor /= 255.0
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor[None]


    # --- Inference ---
    with torch.no_grad():
        pred = model(img_tensor)
        
    # --- Post-processing Steps ---
    # 1. Convert box format from YOLO (cx,cy,w,h) to pixel (x1,y1,x2,y2) using your utility
    # Assuming pred is a list of tensors or a tensor of shape [batch, num_predictions, 5+classes]
    
    box_predictions = Yolo2Pix(pred[..., :4]) # Convert YOLO format label --> pixel label
    other_predictions = pred[..., 4:]
    pred_pixel_coords = torch.cat((box_predictions, other_predictions), dim=-1)

    # 2. Apply Non-Maximum Suppression using your custom function
    ''' Confident + IOU thresh are slow - 10/11 '''

    pred_nms = non_max_suppression(pred_pixel_coords, confidence_threshold=0.25, iou_threshold=0.45)

    # --- Process and Save Final Detections ---
    output_dir = "/home/dangnam/Nam_YOLO11/model_builder/output/YOLO11_Infer"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    txt_path = os.path.join(output_dir, f"{base_name}.txt")
    img_save_path = os.path.join(output_dir, f"{base_name}_predicted.jpg")

    # Convert original PIL image to NumPy array for visualization
    img0_np = np.array(img0)
    # Ensure img0_np is in BGR format for OpenCV visualization
    img0_np = cv2.cvtColor(img0_np, cv2.COLOR_RGB2BGR)


    with open(txt_path, 'w') as f:
        for i, det in enumerate(pred_nms):  # Iterate over detections for each image
            if len(det):
                # Rescale boxes from resized image size back to original image size
                # det[:, :4] = scale_boxes(img.shape[:2], det[:, :4], img0_np.shape, ratio_pad=(ratio, padding_values))
                det[:, :4] = scale_boxes(resized_shape, det[:, :4], img0_np.shape, ratio_pad=(ratio, padding_values))

                # Write results to file and visualize
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    
                    points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    line = ",".join([str(p) for point in points for p in point]) + "\n"
                    f.write(line)

                    # Visualize on the original image
                    label = f'{config["model"]["class_names"][int(cls)]} {conf:.2f}'
                    cv2.polylines(img0_np, [np.array(points, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
                    cv2.putText(img0_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save the visualized image
    cv2.imwrite(img_save_path, img0_np)
    print(f"Predictions saved to {txt_path}")
    print(f"Visualized image saved to {img_save_path}")

if __name__ == '__main__':
    
    # Get only config["Model"] & config ["input_dim"]
    CONFIG_PATH = Path(__file__).parent.parent / "config" / "YOLO11_config.yaml" 
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)


    IMAGE_PATH = "/home/dangnam/Nam_YOLO11/model_builder/inference/test.jpg"
    MODEL_WEIGHTS_PATH = "/home/dangnam/Nam_Yolo11_Text_141025/model_builder/checkpoints/cp_YOLO11n_loss2.9_10-later-pretrained/best.pt"
    # Run the inference
    infer(IMAGE_PATH, MODEL_WEIGHTS_PATH, config)