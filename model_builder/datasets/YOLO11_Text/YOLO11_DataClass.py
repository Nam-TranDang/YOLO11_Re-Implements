import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
import logging # Write log 
from model_builder.utils.YOLO11_Img_Utils import Letterbox_resize
from model_builder.preprocessing.YOLO11_Augment import Random_perspective, Augment_hsv, Albumentations
from model_builder.preprocessing.ConvertPix2Yolo import Pix2Yolo


class YOLO11_Text_Dataset(Dataset):
    def __init__(self, dir_path, transform=None, augment=False, input_size=(480, 480), augment_params=None):
        self.dir_paths = dir_path if isinstance(dir_path, list) else [dir_path]
        self.transform = transform
        self.augment = augment
        self.input_size = input_size
        # Init Augmented data  - V0  not augmented already.  
        self.augment_params = augment_params or {}
        self.albumentations = Albumentations(augment_params) if augment else None
        
        # List data
        self.samples = []
    
        for dir_path in self.dir_paths:
            image_dir = os.path.join(dir_path, 'img')
            box_dir = os.path.join(dir_path, 'box')

            if not os.path.exists(image_dir):
                print(f"Warning: Image directory {image_dir} does not exist. Skipping.")
                continue
            if not os.path.exists(box_dir):
                print(f"Warning: Box directory {box_dir} does not exist. Skipping.")
                continue

            # Support image with format: 
            image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
 
            # Check name label & image name:
            for img_file in image_files:
                box_file = os.path.splitext(img_file)[0] + '.txt'
                box_path = os.path.join(box_dir, box_file)
                if os.path.exists(box_path):
                    self.samples.append((os.path.join(image_dir, img_file), box_path))
                else:
                    print(f"Warning: No box file found for {img_file} in {box_dir}. Skipping.")

        if not self.samples:
            print(f"Warning: No valid image-box pairs found across all directories.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, box_path = self.samples[idx]
        image = Image.open(image_path).convert('RGB')

        # Letterbox resize
        '''
            Cần check self.input_size --> cái này phải đọc từ config 
        '''
        new_image, ratio, (padding_x, padding_y) = Letterbox_resize(image, self.input_size)

        # Load and scale quadrilateral annotations to normalized
        boxes = []
        try:
            with open(box_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    print(f"Warning: Insufficient coordinates in {box_path}: expected at least 8, got {len(parts)}. Skipping line.")
                    continue
                coords = parts[:8]
                coords = list(map(float, coords))

                # Scale to pixel level - match with resized image dimensions
                scaled_coords = [
                    coords[0] * ratio + padding_x,
                    coords[1] * ratio + padding_y,
                    coords[2] * ratio + padding_x,
                    coords[3] * ratio + padding_y,
                    coords[4] * ratio + padding_x,
                    coords[5] * ratio + padding_y,
                    coords[6] * ratio + padding_x,
                    coords[7] * ratio + padding_y
                ]
                boxes.append(scaled_coords)


        except Exception as e:
            print(f"Error reading {box_path}: {e}. Returning empty boxes.")
            boxes = []      

        # Format BBox 
        if len(boxes) == 0:
            boxes_np = np.zeros((0, 5))
        else:
            boxes = np.array(boxes) 
            xs = boxes[:, 0::2]      
            ys = boxes[:, 1::2]      
            x_min = np.min(xs, axis=1)
            y_min = np.min(ys, axis=1)
            x_max = np.max(xs, axis=1)
            y_max = np.max(ys, axis=1)

            bboxes = np.stack((x_min, y_min, x_max, y_max), axis=1) 
            classes = np.zeros((len(bboxes), 1))  
            boxes_np = np.hstack((classes, bboxes))  

        # Convert image to numpy for augmentations
        image_np = np.array(new_image)  # (H, W, C) uint8 RGB

        # Augmentations probabilities
        # if self.augment:
            
        #     # Take Probabilities from YAML file 
        #     prob_persp = self.augment_params.get('prob_perspective', 0.0)
        #     prob_albu = self.augment_params.get('prob_albumentations', 0.0)

        #     rand = np.random.random()  # Single random value in [0,1)
        #     print (rand)
            
        #     # Apply Random Perspective if rand < prob_perspective
        #     if rand < prob_persp:
        #         image_np, boxes_np = Random_perspective(
        #             image_np,
        #             boxes_np,
        #             degrees=self.augment_params.get('degrees', 10.0),
        #             translate=self.augment_params.get('translate', 0.1),
        #             scale=self.augment_params.get('scale', 0.1),
        #             shear=self.augment_params.get('shear', 2.0),
        #             perspective=self.augment_params.get('perspective', 0.0),
        #             border=(0, 0)
        #         )

        #     # Apply Albumentations if rand < prob_albumentations
        #     if rand < prob_albu:
        #         transformed = self.albumentations(image_np, boxes_np[:, 1:], boxes_np[:, 0])
        #         image_np = transformed['image']
        #         if len(transformed['bboxes']) > 0:
        #             boxes_np = np.c_[transformed['class_labels'], transformed['bboxes']]
        #         else:
        #             boxes_np = np.zeros((0, 5))

            # if np.random.random() < prob_hsv:
            #     augment_hsv(
            #         image_np,
            #         hgain=self.augment_params.get('hsv_h', 0.015),
            #         sgain=self.augment_params.get('hsv_s', 0.7),
            #         vgain=self.augment_params.get('hsv_v', 0.4)
            #     )

        # Update height, width after possible augs (though typically size preserved)
        height, width = image_np.shape[:2]

        # Convert to YOLO format: class cx cy w h normalized
        if boxes_np.shape[0] > 0:
            boxes_np[:, [1, 3]] = boxes_np[:, [1, 3]].clip(0, width - 1e-3)      
            boxes_np[:, [2, 4]] = boxes_np[:, [2, 4]].clip(0, height - 1e-3) 
            y = Pix2Yolo(boxes_np, width, height, idx)

            labels = np.hstack((boxes_np[:, :1], y))
            labels = torch.from_numpy(labels).float()  # default return     float32
        else:
            labels = torch.zeros((0, 5), dtype=torch.float32)

        # Convert image back to tensor --> convert from uint8 to float32
        image_tensor = transforms.ToTensor()(image_np)
        
        # label and image both in float 32
        return image_tensor, labels