import torch
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from model import UNet
try:
    from model_advanced import get_model
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
import os

def predict_image(model, image_path, device='cuda'):
    # Read Image (Handle non-ASCII)
    try:
        img_array = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Image decoding failed")
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None, None

    # Preprocess
    h, w = original_image.shape[:2]
    
    # Pad to multiple of 32
    new_h = int(np.ceil(h / 32) * 32)
    new_w = int(np.ceil(w / 32) * 32)
    
    pad_top = 0
    pad_bottom = new_h - h
    pad_left = 0
    pad_right = new_w - w
    
    image_padded = cv2.copyMakeBorder(original_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
    
    input_tensor = torch.from_numpy(image_padded.transpose((2, 0, 1))).float() / 255.0
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
    # Crop back
    pred_mask = pred_mask[:h, :w]
    
    return original_image, pred_mask

def visualize(image, mask, save_path=None):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    # Create a colored mask
    # 0: Background (Black), 1: Crop (Green), 2: Weed (Red)
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 1] = [0, 255, 0] # Green
    colored_mask[mask == 2] = [255, 0, 0] # Red
    
    plt.imshow(colored_mask)
    plt.title('Prediction (Green=Crop, Red=Weed)')
    plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Result saved to {save_path}")
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Path to trained model')
    parser.add_argument('--model_arch', type=str, default='simple_unet', help='Architecture: simple_unet, unet, unetplusplus, deeplabv3plus')
    parser.add_argument('--backbone', type=str, default='resnet34', help='Backbone for advanced models')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--output', type=str, default='result.png', help='Path to save result')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Model file {args.model} not found. Please train the model first.")
        exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    if args.model_arch == 'simple_unet':
        model = UNet(n_channels=3, n_classes=3)
    else:
        if SMP_AVAILABLE:
            model = get_model(args.model_arch, n_classes=3, encoder_name=args.backbone)
        else:
            print("SMP not available, defaulting to simple_unet")
            model = UNet(n_channels=3, n_classes=3)

    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    
    image, mask = predict_image(model, args.image, device)
    
    if image is not None:
        visualize(image, mask, args.output)
