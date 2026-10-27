# python3 -m model_builder.evaluation.eva
import os
import torch
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from model_builder.utils.YOLO11_Metrics_Utils import compute_metric, compute_ap
from model_builder.postprocessing.NonMaxSurpression import non_max_suppression
from model_builder.postprocessing.ConvertYolo2Pix import Yolo2Pix
from model_builder.datasets.YOLO11_Text.YOLO11_DataLoader import test_loader
from model_builder.registry.YOLO11_Registry import Get_Yolo11_Model


'''
Yolo2Pix --> convert tu YOLO sang pix nay hoi dac biet - check lai sau 15/10/2025

'''
def convert_yolo_to_corner(labels, img_width, img_height):
    """
    Convert labels from [class, xc, yc, h, w] to [class, x1, y1, x2, y2].
    
    Args:
        labels (torch.Tensor): Tensor of shape [N, 5] with [class, xc, yc, h, w].
        img_width (int): Image width for scaling.
        img_height (int): Image height for scaling.
    
    Returns:
        torch.Tensor: Converted labels [class, x1, y1, x2, y2].
    """
    converted = labels.clone()
    # Convert normalized center coordinates and dimensions
    converted[:, 0] = (labels[:, 0] - labels[:, 2] / 2) * img_width   # x1 = xc - w/2
    converted[:, 1] = (labels[:, 1] - labels[:, 3] / 2) * img_height  # y1 = yc - h/2
    converted[:, 2] = (labels[:, 0] + labels[:, 2] / 2) * img_width   # x2 = xc + w/2
    converted[:, 3] = (labels[:, 1] + labels[:, 3] / 2) * img_height  # y2 = yc + h/2
    return converted

@torch.no_grad()
def evaluate_yolo11(config, model_weights_path, output_dir):
    """
    Evaluate YOLO11 model on the test dataset, compute metrics, and save results.
    
    Args:
        config (dict): Configuration dictionary from YAML file.
        model_weights_path (str): Path to the trained model weights.
        output_dir (str): Directory to save evaluation results and plots.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model_name = config["model"]["variant"]
    num_classes = config["model"]["nc"]
    model = Get_Yolo11_Model(model_name, num_classes)
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu'))['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Configure IoU thresholds for mAP@0.5:0.95
    iou_v = torch.linspace(0.5, 0.95, 10).to(device)  # IoU vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    # Initialize lists to collect metrics
    metrics = []
    # results_file = os.path.join(output_dir, "evaluation_results.txt")
    results_file = os.path.join(output_dir, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        f.write("YOLO11 Evaluation Results\n")
        f.write("=" * 50 + "\n")

    # Progress bar for evaluation
    p_bar = tqdm(test_loader, desc=('%10s' * 5) % ('', 'Precision', 'Recall', 'mAP50', 'mAP'))

    # for batch_idx, (images, labels) in enumerate(p_bar):    
    for batch_idx, batch in enumerate(p_bar):
        # Debug: Inspect batch
        if isinstance(batch, tuple) and len(batch) == 2:
            images, labels = batch
        else:
            print(f"Batch {batch_idx + 1}: Unexpected batch format: {type(batch)}")
            break
        # images = images.to(device).float() / 255.0  # Normalize to [0, 1]
       
        images = images.to(device)  # Already float [0,1] from dataloader
        _, _, h, w = images.shape  # Batch-size, channels, height, width
        # scale = torch.tensor((w, h, w, h), device=device)

        # Debug Label load
        # print(f"Type of labels: {type(labels)}")    # Class <'list'>
        # print(f"Targets content: {labels}")         # tensor for each img with (class, xc, yc, h, w)
        # break

        # Inference
        outputs = model(images)
        # print(f"Batch {batch_idx + 1} - Images shape: {images.shape}, dtype: {images.dtype}")
        
        
        if not isinstance(labels, list):
            print(f"Batch {batch_idx + 1}: Expected labels to be a list of tensors, got {type(labels)}")
            with open(results_file, 'a') as f:
                f.write(f"Batch {batch_idx + 1}: Invalid labels format: {type(labels)}\n")
            break

        # Debug: Log label details
        for i, lbl in enumerate(labels):
            print(f"Batch {batch_idx + 1}, Image {i} - Labels shape: {lbl.shape}, dtype: {lbl.dtype}")
            break
        # if isinstance(labels, dict) and 'cls' in labels and 'box' in labels and 'idx' in labels:
        #     print(f"Batch {batch_idx + 1} - Targets: cls shape: {labels['cls'].shape}, "
        #                f"box shape: {labels['box'].shape}, idx shape: {labels['idx'].shape}")
        # else:
        #     print(f"Batch {batch_idx + 1}: Invalid target format: {labels}")
        #     break
        # Apply Non-Max Suppression
        outputs = non_max_suppression(outputs, confidence_threshold=0.001, iou_threshold=0.65)
        # print(f"Batch {batch_idx + 1} - Model outputs shape: {outputs}")

        # Process each image in the batch
        for i, (output, label) in enumerate(zip(outputs, labels)):
            # Move labels to device
            label = label.to(device)
            cls = label[:, 0:1]  # [N, 1]
            box = label[:, 1:5]  # [N, 4] -> [xc, yc, h, w] (normalized)

            if output.shape[0] == 0:
                if cls.shape[0]:
                    # No detections but there are ground truths
                    metrics.append((torch.zeros((0, n_iou), dtype=torch.bool, device=device),
                                    torch.zeros(0, device=device),
                                    torch.zeros(0, device=device),
                                    cls.squeeze(-1)))
                continue
            
            # Clip output boxes
            img_size = torch.tensor([w, h, w, h], device=device)
            output[:, 0] = torch.clamp(output[:, 0], 0, w)
            output[:, 1] = torch.clamp(output[:, 1], 0, h)
            output[:, 2] = torch.clamp(output[:, 2], 0, w)
            output[:, 3] = torch.clamp(output[:, 3], 0, h)
            output[:, 2] = torch.max(output[:, 2], output[:, 0])
            output[:, 3] = torch.max(output[:, 3], output[:, 1])
            
            # Convert ground truth boxes to (x1, y1, x2, y2)
            if cls.shape[0]:

                box_converted = convert_yolo_to_corner(box, w, h)  # [N, 4] pixel [x1,y1,x2,y2]
                target = torch.cat((cls, box_converted), dim=1)    # [N, 5] [class, x1,y1,x2,y2] pixel
                metric = compute_metric(output[:, :6], target, iou_v)                
                # print (f"compute metric {metric}")

                metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))

                # target = convert_yolo_to_corner(box, w, h)  # Use custom conversion --> w, h là của ảnh thật 
                # print(f"Batch {batch_idx + 1}, Image {i} - Target shape: {target.shape}, sample: {target[:3]}")
                # metric = compute_metric(output[:, :6], target, iou_v)
                # print (f"compute metric {metric}")

                
            else:
                print(f"Batch {batch_idx + 1}, Image {i}: No ground truth labels")

        # Compute metrics for the batch
        if metrics:
            metrics_np = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)]
            if len(metrics_np) and metrics_np[0].any():
                tp, fp, m_pre, m_rec, map50, mean_ap = compute_ap(
                    *metrics_np, plot=True, names=config["model"]["names"], save_dir=output_dir
                )
                
                print ( tp, fp, m_pre, m_rec, map50, mean_ap )

                # Update progress bar
                p_bar.set_description(('%10s' + '%10.3g' * 4) % ('', m_pre, m_rec, map50, mean_ap))

                # Save results to text file
                with open(results_file, 'a') as f:
                    f.write(f"Batch {batch_idx + 1}:\n")
                    f.write(f"Precision: {m_pre:.3f}\n")
                    f.write(f"Recall: {m_rec:.3f}\n")
                    f.write(f"mAP@0.5: {map50:.3f}\n")
                    f.write(f"mAP@0.5:0.95: {mean_ap:.3f}\n")
                    f.write(f"True Positives: {tp.sum()}\n")
                    f.write(f"False Positives: {fp.sum()}\n")
                    f.write("-" * 50 + "\n")

    # Compute and save final metrics
    if metrics:
        metrics_np = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)]
        if len(metrics_np) and metrics_np[0].any():
            tp, fp, m_pre, m_rec, map50, mean_ap = compute_ap(
                *metrics_np, plot=True, names=config["model"]["names"], save_dir=output_dir
            )

            # Save final results
            with open(results_file, 'a') as f:
                f.write("Final Results:\n")
                f.write(f"Precision: {m_pre:.3f}\n")
                f.write(f"Recall: {m_rec:.3f}\n")
                f.write(f"mAP@0.5: {map50:.3f}\n")
                f.write(f"mAP@0.5:0.95: {mean_ap:.3f}\n")
                f.write(f"True Positives: {tp.sum()}\n")
                f.write(f"False Positives: {fp.sum()}\n")
                f.write("-" * 50 + "\n")

            print(f"Evaluation completed. Results saved to {results_file}")
            print(f"Plots saved to {output_dir}")
        else:
            print("No valid metrics computed.")
    else:
        print("No metrics collected during evaluation.")


    ''' Note here '''
    # Return model to float for training if needed
    model.float()

def main():
    # Read config
    CONFIG_PATH = Path(__file__).parent.parent / "config" / "YOLO11_config.yaml"
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    MODEL_WEIGHTS_PATH = "~/Nam_YOLO11/model_builder/checkpoints/YOLO11_Checkpoint/best.pt"
    OUTPUT_EVAL = "~/Nam_YOLO11/model_builder/output/YOLO11_Eval_Result"

    weight_path = os.path.expanduser(MODEL_WEIGHTS_PATH)
    output_eval_path = os.path.expanduser(OUTPUT_EVAL)

    # Ensure output directory exists
    os.makedirs(output_eval_path, exist_ok=True)

    # Evaluate
    evaluate_yolo11(config, weight_path, output_eval_path)

if __name__ == "__main__":
    main()