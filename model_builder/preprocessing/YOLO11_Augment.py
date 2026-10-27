import math
import random
import cv2
import numpy as np

try:
    import albumentations as A
except ImportError:
    A = None

'''
Note: Augment technique reference from 
https://github.com/jahongir7174/YOLOv11-pt/blob/master/utils/dataset.py
'''


def Candidates(box1, box2):
    # box1(4,n), box2(4,n)
    w1 = box1[2] - box1[0]
    h1 = box1[3] - box1[1]

    w2 = box2[2] - box2[0]
    h2 = box2[3] - box2[1]
    aspect_ratio = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))     # aspect ratio
    return (w2 > 2) & (h2 > 2) & (w2 * h2 / (w1 * h1 + 1e-16) > 0.1) & (aspect_ratio < 100)


# Augment rotation
def Random_perspective(image, labels, degrees=10.0, translate=0.1, scale=0.1, shear=2.0, perspective=0.0, border=(0, 0)):
    height = image.shape[0] + border[0] * 2
    width = image.shape[1] + border[1] * 2

    # Center
    Center = np.eye(3)
    Center[0, 2] = -image.shape[1] / 2
    Center[1, 2] = -image.shape[0] / 2

    # Perspective (not used if 0)
    Perspective = np.eye(3)
    Perspective[2, 0] = random.uniform(-perspective, perspective)
    Perspective[2, 1] = random.uniform(-perspective, perspective)

    # Rotation and Scale
    Rotate = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    Rotate[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    Shear = np.eye(3)
    Shear[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    Shear[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    # Translation
    Translate = np.eye(3)
    Translate[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    Translate[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

    # Combined matrix
    Matrix = Translate @ Shear @ Rotate @ Perspective @ Center
    if (border[0] != 0) or (border[1] != 0) or (Matrix != np.eye(3)).any():
        image = cv2.warpAffine(image, Matrix[:2], dsize=(width, height), borderValue=(0, 0, 0))

    # Transform labels
    n = len(labels)
    if n:
        # For quads approximated to rects, but since we use min-max, transform the 4 corners
        xy = np.ones((n * 4, 3))
        # Assume labels[:,1:5] is x1 y1 x2 y2, but to handle potential rotation, use corners
        xy[:, :2] = labels[:, [1, 2, 3, 2, 3, 4, 1, 4]].reshape(n * 4, 2)  # x1y1, x2y1, x2y2, x1y2
        xy = (xy @ Matrix.T)[:, :2].reshape(n, 8)

        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        i = Candidates(labels[:, 1:5].T, new.T)
        labels = labels[i]
        labels[:, 1:5] = new[i]

    return image, labels

# Augmented change color 
def Augment_hsv(image, hue_gain=0.015, satur_gain=0.7, value_gain=0.4):
    if hue_gain or satur_gain or value_gain:
        r = np.random.uniform(-1, 1, 3) * [hue_gain, satur_gain, value_gain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))

        dtype = image.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB, dst=image)

# Augmented (blur, noise, bright)
class Albumentations:
    def __init__(self, params):
        self.transform = None
        if A is not None:
            self.transform = A.Compose([
                A.Blur(blur_limit=params.get('blur_limit', 3), p=0.5),
                A.CLAHE(clip_limit=params.get('clahe_clip_limit', 2.0), p=0.5),
                A.ToGray(p=0.5),
                A.GaussNoise(var_limit=(0.0, params.get('noise_stddev', 25.0)**2), p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=params.get('brightness_contrast_limit', 0.2),
                    contrast_limit=params.get('brightness_contrast_limit', 0.2),
                    p=0.5
                ),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def __call__(self, image, bboxes, class_labels):
        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            return transformed
        return {'image': image, 'bboxes': bboxes, 'class_labels': class_labels}