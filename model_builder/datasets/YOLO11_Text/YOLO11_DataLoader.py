# cli run Module to test: python3 -m .datasets.YOLO11_Text.YOLO11_DataLoader
import argparse
import yaml
import os
from pathlib import Path
import logging
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from model_builder.datasets.YOLO11_Text.YOLO11_DataClass import YOLO11_Text_Dataset
from model_builder.datasets.YOLO11_Text.YOLO11_Collate import YOLO11_collate_fn

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "YOLO11_config.yaml" 
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


# Allow to use ~ as home
root_dir = os.path.expanduser(config["root_dir"])

# Function check path & read path 
def get_valid_paths(path_items, root_dir, dataset_type):
    valid_paths = []
    try:
        for item in path_items:
            path = item["path"]
            full_path = os.path.join(root_dir, path.lstrip('/'))
            if os.path.exists(full_path):
                valid_paths.append(full_path)
            else:
                print(f"Warning: {dataset_type} path does not exist and will be ignored: {full_path}")
    except (KeyError, TypeError) as e:
        print(f"Warning: Invalid {dataset_type} configuration in YAML: {e}")
    return valid_paths

train_paths = get_valid_paths(config.get("data_dirs", {}).get("train", []), root_dir, "Train")
val_paths = get_valid_paths(config.get("data_dirs", {}).get("validation", []), root_dir, "Validation")
test_paths = get_valid_paths(config.get("data_dirs", {}).get("test", []), root_dir, "Test")

# Some params for training 
transform = transforms.ToTensor()  # Used after augmentations

batch_size = config["training"]["batch_size"]
num_workers = config["training"]["num_workers"]
input_size = (config["input_dim"]["height"], config["input_dim"]["width"])
augment_params = config["augmentations"]

# Read all image from path --> return an "Object"
train_datasets = [YOLO11_Text_Dataset(path, transform=transform, augment=True, input_size=input_size, augment_params=augment_params) for path in train_paths]
val_datasets = [YOLO11_Text_Dataset(path, transform=transform, augment=False, input_size=input_size, augment_params=augment_params) for path in val_paths]
test_datasets = [YOLO11_Text_Dataset(path, transform=transform, augment=False, input_size=input_size, augment_params=augment_params) for path in test_paths]

# Connect params
train_dataset_con = ConcatDataset(train_datasets) if train_datasets else None
val_dataset_con   = ConcatDataset(val_datasets) if val_datasets else None
test_dataset_con  = ConcatDataset(test_datasets) if test_datasets else None

# Load data 
train_loader = DataLoader(train_dataset_con, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=YOLO11_collate_fn) if train_dataset_con else None
val_loader = DataLoader(val_dataset_con, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=YOLO11_collate_fn) if val_dataset_con else None
test_loader = DataLoader(test_dataset_con, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=YOLO11_collate_fn) if test_dataset_con else None

if train_loader is None:
    print("Warning: Train dataset is empty. Check paths in config.yaml.")
if val_loader is None:
    print("Warning: Validation dataset is empty. Skipping validation.")
if test_loader is None:
    print("Warning: Test dataset is empty. Skipping testing.")