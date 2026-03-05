import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from dataset import WeedDataset
from model_advanced import get_model
from tqdm import tqdm
import os
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

def calculate_metrics_batch(pred_mask, true_mask, num_classes=3):
    # pred_mask: [B, H, W]
    # true_mask: [B, H, W]
    
    # Flatten
    pred = pred_mask.view(-1).cpu().numpy()
    true = true_mask.view(-1).cpu().numpy()
    
    # Calculate Precision, Recall, F1 per class
    # labels=[0, 1, 2] ensures we get metrics for all classes even if missing in batch
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=list(range(num_classes)), average=None, zero_division=0
    )
    
    # IoU calculation
    iou_list = []
    for cls in range(num_classes):
        intersection = ((pred == cls) & (true == cls)).sum()
        union = ((pred == cls) | (true == cls)).sum()
        if union == 0:
            iou_list.append(float('nan'))
        else:
            iou_list.append(intersection / union)
            
    return precision, recall, f1, iou_list

def tta_inference(model, images):
    """
    Test-Time Augmentation:
    1. Original
    2. Horizontal Flip
    3. Vertical Flip
    """
    # 1. Original
    out_orig = model(images)
    probs = F.softmax(out_orig, dim=1)
    
    # 2. Horizontal Flip
    images_h = torch.flip(images, dims=[3])
    out_h = model(images_h)
    probs_h = F.softmax(out_h, dim=1)
    probs_h = torch.flip(probs_h, dims=[3])
    probs += probs_h
    
    # 3. Vertical Flip
    images_v = torch.flip(images, dims=[2])
    out_v = model(images_v)
    probs_v = F.softmax(out_v, dim=1)
    probs_v = torch.flip(probs_v, dims=[2])
    probs += probs_v
    
    # Average
    probs /= 3.0
    return probs

import time

def evaluate(model_path, data_dir, model_name='manet', backbone='efficientnet-b4', use_tta=True, batch_size=4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Evaluating model: {model_path}")
    print(f"Method: {model_name} + {backbone}")
    print(f"TTA Enabled: {use_tta}")
    
    dataset = WeedDataset(data_dir, mode='val', augment=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Load Model
    model = get_model(model_name, encoder_name=backbone, n_classes=3)
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Weights loaded successfully.")
    else:
        print(f"Error: Model file not found at {model_path}")
        return

    model = model.to(device)
    model.eval()
    
    # Metrics aggregators
    metrics = {
        'iou': [[] for _ in range(3)],
        'precision': [[] for _ in range(3)],
        'recall': [[] for _ in range(3)],
        'f1': [[] for _ in range(3)]
    }
    
    total_time = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images, masks = batch
            images = images.to(device)
            
            start_time = time.time()
            if use_tta:
                probs = tta_inference(model, images)
                pred_mask = torch.argmax(probs, dim=1)
            else:
                outputs = model(images)
                pred_mask = torch.argmax(outputs, dim=1)
            end_time = time.time()
            
            batch_time = end_time - start_time
            total_time += batch_time
            total_samples += images.size(0)
            
            precision, recall, f1, iou_list = calculate_metrics_batch(pred_mask, masks, num_classes=3)
            
            for cls_idx in range(3):
                metrics['precision'][cls_idx].append(precision[cls_idx])
                metrics['recall'][cls_idx].append(recall[cls_idx])
                metrics['f1'][cls_idx].append(f1[cls_idx])
                if not np.isnan(iou_list[cls_idx]):
                    metrics['iou'][cls_idx].append(iou_list[cls_idx])
    
    # Calculate FPS
    fps = total_samples / total_time if total_time > 0 else 0

    print("\n" + "="*50)
    print(f"Evaluation Report: {os.path.basename(model_path)}")
    print(f"Method: {model_name} + {backbone}")
    print(f"TTA: {'Enabled' if use_tta else 'Disabled'}")
    print(f"Inference Speed: {fps:.2f} FPS (Batch Size: {batch_size})")
    print("="*50)
    
    classes = {0: 'Background', 1: 'Crop', 2: 'Weed'}
    
    # Print per-class metrics
    print(f"{'Class':<15} | {'IoU':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 65)
    
    m_ious = []
    
    for cls_idx in range(3):
        iou = np.mean(metrics['iou'][cls_idx]) if metrics['iou'][cls_idx] else 0
        prec = np.mean(metrics['precision'][cls_idx])
        rec = np.mean(metrics['recall'][cls_idx])
        f1_score = np.mean(metrics['f1'][cls_idx])
        
        m_ious.append(iou)
        
        print(f"{classes[cls_idx]:<15} | {iou:.4f}     | {prec:.4f}     | {rec:.4f}     | {f1_score:.4f}")
    
    print("-" * 65)
    print(f"{'Mean (Macro)':<15} | {np.mean(m_ious):.4f}     | -          | -          | -")
    print("="*50)
    
    return np.mean(m_ious)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default=r"c:\Users\lsy\Desktop\温室无人机巡检系统\杂草\Tobacco Aerial Dataset")
    parser.add_argument('--model', type=str, default='manet')
    parser.add_argument('--backbone', type=str, default='efficientnet-b4')
    parser.add_argument('--no_tta', action='store_true', help='Disable Test-Time Augmentation')
    
    args = parser.parse_args()
    
    evaluate(args.model_path, args.data_dir, args.model, args.backbone, not args.no_tta)
