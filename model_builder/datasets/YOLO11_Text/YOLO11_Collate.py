import torch

def YOLO11_collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, labels