import yaml
from pathlib import Path
import torch
import os

from .registry.YOLO11_Registry import Get_Yolo11_Model
from .registry.Optimizer_Registry import Get_Optimizer                  # Get Optim Register
from .registry.Loss_Registry import Get_Loss                            # Get Loss Register
from .datasets.YOLO11_Text.YOLO11_DataLoader import train_loader

# from .datasets.YOLO11_COCO17.YOLO11_COCO17_DataLoader import train_loader # Version to load only COCO17 dataset

from  .training.YOLO11_Optimizer import EMA, set_params, clip_gradients # Get Optim training technique
from .utils.YOLO11_Train_Utils import setup_seed, setup_multi_processes, AverageMeter, plot_lr  # For loss dependencies if needed
''' This utils have visulize code'''

CONFIG_PATH = Path(__file__).parent.parent / "config" / "YOLO11_config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# a. Training params~~
epochs = config["training"]['epochs']
save_dir = config["training"]["save_dir"]
save_dir = os.path.abspath(os.path.expanduser(save_dir))


patience = config["training"]["patience"]

# Set up training multi thread + reproduce same result (seed) - Optional
# setup_seed()
# setup_multi_processes()

# b. Get Model from Registry
model_name  = config["model"]["variant"]
num_classes = config["model"]["nc"]
model = Get_Yolo11_Model (model_name, num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Set device Cuda || CPU
model.to(device)

# c. Get Loss 
loss_cfg = config["loss"]
loss_name = loss_cfg["loss_type"]
loss_fn = Get_Loss(loss_name, model = model, params = loss_cfg) # Call và truyền Params vào Computeloss với 

'''
Getloss - take 2 input 
    a/ model name: ComputeLoss 
    b/ config params: box, clf, dfl
'''

# d. Optimizer 
optimizer_cfg = config["optimizer"] 
param_groups = set_params(model, optimizer_cfg["weight_decay"])     # Get Decay at 5x10(-4) for Scheduler 

if optimizer_cfg["optim_type"].lower() == "sgd":
    optimizer = torch.optim.SGD(
        param_groups,
        lr = optimizer_cfg["max_lr"],
        momentum = optimizer_cfg["momentum"]
    )

elif optimizer_cfg["type"].lower() == "adam":
    optimizer = torch.optim.Adam(
        param_groups,
        lr = optimizer_cfg["max_lr"],
        weight_decay = optimizer_cfg["weight_decay"]
    )
else:
    raise ValueError(f"Unsupported optimizer type: {optimizer_cfg["type"]}")



# e. Scheduler (store in registry)
scheduler_type = optimizer_cfg["scheduler"]

if train_loader is None:
    raise ValueError("Train loader is None. Check train paths in config.yaml or dataset initialization.")
num_steps = len(train_loader)       # Call the Num_Steps for Learning Rate -- phu thuoc vao batch size

scheduler = Get_Optimizer(
    scheduler_type,
    epochs = epochs,
    params = optimizer_cfg,
    num_steps = num_steps
)

# f. EMA train
use_ema = optimizer_cfg["use_ema"] 
if use_ema == True:
    ema = EMA(model)
else:
    None

# g. Plot Learning Rate curve
plot_lr(epochs, optimizer, scheduler, num_steps)

# Training loop
best_loss = float('inf')
patience_counter = 0
global_step = 0

for epoch in range(epochs):
    model.train()
    loss_meter = AverageMeter()

    for images, labels in train_loader:
        images = images.to(device)
        # labels is list of tensors [num_boxes, 5] per image in batch

        # Process to targets dict for loss
        all_idx = []
        all_cls = []
        all_box = []
        has_labels = False
        for bi, label in enumerate(labels):
            num = label.shape[0]
            if num > 0:
                has_labels = True
            all_idx.extend([bi] * num)
            all_cls.append(label[:, 0:1])
            all_box.append(label[:, 1:])

        if not has_labels:
            continue  # Skip batch with no labels

        # Group box + class --> use for calculate loss 
        targets = {
            'idx': torch.tensor(all_idx, dtype=torch.float32, device=device),
            'cls': torch.cat(all_cls).to(device),
            'box': torch.cat(all_box).to(device)
        }

        optimizer.zero_grad()

        outputs = model(images) # Pass images for model --> create the result 

        loss_box, loss_cls, loss_dfl = loss_fn(outputs, targets)   # Compute Loss output model (image) + actual target  
        loss = loss_box + loss_cls + loss_dfl

        loss.backward()

        clip_gradients(model)

        optimizer.step()

        scheduler.step(global_step, optimizer)

        if use_ema:
            ema.update(model)

        loss_meter.update(loss.item(), images.size(0))

        global_step += 1

    avg_loss = loss_meter.avg
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    # Early stopping based on training loss (since test dir is commented; add val if needed)
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save({'model': model.state_dict()}, os.path.join(save_dir, "best.pt"))
        
        if use_ema:
            torch.save({'model': ema.ema.state_dict()}, f"{save_dir}/ema_best.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

print("Training complete.")