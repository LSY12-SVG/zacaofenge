import os
import subprocess
import argparse
import sys
import shutil

def run_command(command, log_file=None):
    print(f"\nRunning: {command}")
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n\nCommand: {command}\n")
            f.write("="*50 + "\n")
            
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors='replace')
            
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    print(line.strip())
                    f.write(line)
    else:
        ret = os.system(command)
        return ret == 0
    return True

def run_experiments(data_dir):
    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    report_file = "results/paper_experiment_report.txt"
    if os.path.exists(report_file):
        os.remove(report_file)
        
    print(f"Results will be saved to {report_file}")

    # Best Model Path (Current Best)
    best_model_path = "best_model_manet_optimized_continued.pth"
    if not os.path.exists(best_model_path):
        print(f"Warning: {best_model_path} not found in root. Checking models/ directory...")
        if os.path.exists(f"models/{best_model_path}"):
            best_model_path = f"models/{best_model_path}"
        else:
            print("Error: Best model not found. Please train it first.")
            # return

    # 1. Comparative Experiments
    # Exp-1: UNet Baseline (ResNet34)
    print("\n=== Exp-1: UNet Baseline (Training) ===")
    unet_path = "models/unet_baseline.pth"
    if not os.path.exists(unet_path):
        print("Training UNet Baseline...")
        # Train for 30 epochs as baseline
        run_command(f"python src/train.py --model unet --backbone resnet34 --epochs 30 --save_name {unet_path} --data_dir \"{data_dir}\"", log_file=report_file)
    else:
        print("UNet Baseline already exists. Skipping training.")

    # 2. Evaluation
    print("\n=== Evaluation Phase ===")
    
    # Eval 1: UNet Baseline
    print("\n--- Evaluating UNet Baseline ---")
    run_command(f"python src/evaluate.py --model_path {unet_path} --model unet --backbone resnet34 --data_dir \"{data_dir}\" --no_tta", log_file=report_file)
    
    # Eval 2: Proposed Method (MAnet + TTA)
    print("\n--- Evaluating Proposed Method (MAnet + TTA) ---")
    if os.path.exists(best_model_path):
        run_command(f"python src/evaluate.py --model_path {best_model_path} --model manet --backbone efficientnet-b4 --data_dir \"{data_dir}\"", log_file=report_file)
    
    # Eval 3: Ablation (MAnet No TTA)
    print("\n--- Evaluating Ablation (MAnet No TTA) ---")
    if os.path.exists(best_model_path):
        run_command(f"python src/evaluate.py --model_path {best_model_path} --model manet --backbone efficientnet-b4 --data_dir \"{data_dir}\" --no_tta", log_file=report_file)

    # 3. Visualization
    print("\n=== Visualization Phase ===")
    if os.path.exists(best_model_path):
        viz_save_path = "results/paper_visualization_comparison.png"
        run_command(f"python src/visualize.py --model_path {best_model_path} --save_path {viz_save_path} --data_dir \"{data_dir}\"")
        print(f"Visualization saved to {viz_save_path}")

    print(f"\nAll experiments completed. Check {report_file} for detailed metrics.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default to the known absolute path
    parser.add_argument('--data_dir', type=str, default=r"c:\Users\lsy\Desktop\温室无人机巡检系统\杂草\Tobacco Aerial Dataset")
    args = parser.parse_args()
    
    run_experiments(args.data_dir)
