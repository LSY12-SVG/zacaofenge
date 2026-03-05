import torch
import numpy as np
import cv2
import os
import argparse
from torch.utils.data import DataLoader
from dataset import WeedDataset
from model_advanced import get_model
from tqdm import tqdm
import matplotlib.pyplot as plt

def visualize_comparison(model_path, data_dir, save_path='paper_viz.png', num_samples=5, model_name='manet', backbone='efficientnet-b4'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load Dataset (Validation)
    dataset = WeedDataset(data_dir, mode='val', augment=False)
    
    # 1. Scan for interesting samples (High Weed Density)
    # This is similar to visualize_smart.py logic but consolidated here
    print("Scanning for high-density weed samples...")
    weed_counts = []
    
    # Limit scan to first 500 samples to be fast, or scan all if needed.
    # Since we have pre-computed masks on disk, we can read them fast.
    # dataset.masks is a list of paths
    for i, mask_path in enumerate(tqdm(dataset.masks)):
        # Fast read using OpenCV with unicode support fix
        try:
            mask_array = np.fromfile(mask_path, dtype=np.uint8)
            mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
            if mask is None: continue
        except Exception:
            continue
        weed_pixel_count = np.sum(mask == 2) # Class 2 is weed
        if weed_pixel_count > 0:
            weed_counts.append((i, weed_pixel_count))
            
    # Sort by weed count descending
    weed_counts.sort(key=lambda x: x[1], reverse=True)
    
    # Select top N samples
    selected_indices = [x[0] for x in weed_counts[:num_samples]]
    print(f"Selected {len(selected_indices)} samples for visualization.")
    
    # Load Model
    model = get_model(model_name, encoder_name=backbone, n_classes=3)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Model not found: {model_path}")
        return

    model.to(device)
    model.eval()
    
    # Setup Plot
    # Rows = num_samples, Cols = 3 (Input, Ground Truth, Prediction)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    colors = [
        [0, 0, 0],       # 0: Background (Black)
        [0, 255, 0],     # 1: Crop (Green)
        [255, 0, 0]      # 2: Weed (Red)
    ]
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(selected_indices):
            image, mask = dataset[sample_idx] # image is tensor, mask is tensor
            
            # Prepare Input for Plot (CHW -> HWC)
            img_disp = image.permute(1, 2, 0).cpu().numpy()
            # De-normalize if needed (assuming dataset outputs normalized tensor)
            # Simple Min-Max for display
            img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min())
            
            # Predict
            input_tensor = image.unsqueeze(0).to(device)
            output = model(input_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # Prepare GT
            gt_mask = mask.cpu().numpy()
            
            # Colorize Masks
            def colorize(m):
                h, w = m.shape
                c_img = np.zeros((h, w, 3), dtype=np.uint8)
                for cls in range(3):
                    c_img[m == cls] = colors[cls]
                return c_img
            
            gt_color = colorize(gt_mask)
            pred_color = colorize(pred_mask)
            
            # Plot
            ax_row = axes[idx] if num_samples > 1 else axes
            
            # Input
            ax_row[0].imshow(img_disp)
            ax_row[0].set_title("Input Image")
            ax_row[0].axis('off')
            
            # GT
            ax_row[1].imshow(gt_color)
            ax_row[1].set_title("Ground Truth")
            ax_row[1].axis('off')
            
            # Pred
            ax_row[2].imshow(pred_color)
            ax_row[2].set_title("Prediction")
            ax_row[2].axis('off')
            
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default=r"c:\Users\lsy\Desktop\温室无人机巡检系统\杂草\Tobacco Aerial Dataset")
    parser.add_argument('--save_path', type=str, default='experiment_viz.png')
    
    args = parser.parse_args()
    visualize_comparison(args.model_path, args.data_dir, args.save_path)
