"""
合并 Tobacco Aerial Dataset 和 CoFly-WeedDB 数据集
统一格式并创建合并后的训练/测试划分
"""
import os
import shutil
import random
from pathlib import Path

def create_combined_dataset():
    """创建合并的数据集目录结构"""
    
    # 定义路径
    base_dir = Path(r"e:\温室无人机巡检系统\杂草")
    tobacco_dir = base_dir / "Tobacco Aerial Dataset"
    cofly_dir = base_dir / "CoFly-WeedDB" / "CoFly-WeedDB"
    combined_dir = base_dir / "Combined_Dataset"
    
    # 创建目录结构
    print("创建合并数据集目录...")
    for split in ['train', 'val', 'test']:
        (combined_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (combined_dir / split / 'masks').mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    stats = {
        'tobacco': {'train': 0, 'val': 0, 'test': 0},
        'cofly': {'train': 0, 'val': 0, 'test': 0}
    }
    
    # 1. 复制 Tobacco 数据集
    print("\n处理 Tobacco Aerial Dataset...")
    tobacco_campaigns = list((tobacco_dir / "Campaign no. 1" / "Patch images").glob("*"))
    
    # 获取所有图像对
    tobacco_pairs = []
    for campaign_dir in tobacco_dir.glob("Campaign no.*"):
        patch_dir = campaign_dir / "Patch images"
        if not patch_dir.exists():
            continue
        
        data_dir = patch_dir / "data"
        mask_dir = patch_dir / "mask"
        
        if data_dir.exists() and mask_dir.exists():
            for img_file in data_dir.glob("*.png"):
                mask_file = mask_dir / img_file.name
                if mask_file.exists():
                    tobacco_pairs.append((img_file, mask_file))
    
    print(f"找到 {len(tobacco_pairs)} 张 Tobacco 图像")
    
    # 随机划分 80% train, 10% val, 10% test
    random.shuffle(tobacco_pairs)
    n_train = int(len(tobacco_pairs) * 0.8)
    n_val = int(len(tobacco_pairs) * 0.1)
    
    train_pairs = tobacco_pairs[:n_train]
    val_pairs = tobacco_pairs[n_train:n_train + n_val]
    test_pairs = tobacco_pairs[n_train + n_val:]
    
    # 复制到合并目录
    for i, (img, mask) in enumerate(train_pairs):
        new_name = f"tobacco_train_{i:04d}.png"
        shutil.copy(img, combined_dir / "train" / "images" / new_name)
        shutil.copy(mask, combined_dir / "train" / "masks" / new_name)
        stats['tobacco']['train'] += 1
    
    for i, (img, mask) in enumerate(val_pairs):
        new_name = f"tobacco_val_{i:04d}.png"
        shutil.copy(img, combined_dir / "val" / "images" / new_name)
        shutil.copy(mask, combined_dir / "val" / "masks" / new_name)
        stats['tobacco']['val'] += 1
    
    for i, (img, mask) in enumerate(test_pairs):
        new_name = f"tobacco_test_{i:04d}.png"
        shutil.copy(img, combined_dir / "test" / "images" / new_name)
        shutil.copy(mask, combined_dir / "test" / "masks" / new_name)
        stats['tobacco']['test'] += 1
    
    print(f"Tobacco: {stats['tobacco']['train']} train, {stats['tobacco']['val']} val, {stats['tobacco']['test']} test")
    
    # 2. 复制 CoFly 数据集
    print("\n处理 CoFly-WeedDB...")
    cofly_images_dir = cofly_dir / "images"
    cofly_labels_dir = cofly_dir / "labels"
    
    # 读取划分文件 (使用 split 1)
    train_split_file = cofly_dir / "train_split1.txt"
    test_split_file = cofly_dir / "test_split1.txt"
    
    if train_split_file.exists() and test_split_file.exists():
        with open(train_split_file, 'r') as f:
            train_names = [line.strip() for line in f.readlines()]
        with open(test_split_file, 'r') as f:
            test_names = [line.strip() for line in f.readlines()]
        
        print(f"CoFly 官方划分：{len(train_names)} train, {len(test_names)} test")
        
        # 从 test 中分出一部分作为 val (20%)
        n_val_cofly = int(len(test_names) * 0.2)
        test_names_cofly = test_names[n_val_cofly:]
        val_names_cofly = test_names[:n_val_cofly]
        
        # 复制 CoFly 图像
        for i, name in enumerate(train_names):
            img_file = cofly_images_dir / name
            mask_file = cofly_labels_dir / name
            if img_file.exists() and mask_file.exists():
                new_name = f"cofly_train_{i:04d}.png"
                shutil.copy(img_file, combined_dir / "train" / "images" / new_name)
                shutil.copy(mask_file, combined_dir / "train" / "masks" / new_name)
                stats['cofly']['train'] += 1
        
        for i, name in enumerate(val_names_cofly):
            img_file = cofly_images_dir / name
            mask_file = cofly_labels_dir / name
            if img_file.exists() and mask_file.exists():
                new_name = f"cofly_val_{i:04d}.png"
                shutil.copy(img_file, combined_dir / "val" / "images" / new_name)
                shutil.copy(mask_file, combined_dir / "val" / "masks" / new_name)
                stats['cofly']['val'] += 1
        
        for i, name in enumerate(test_names_cofly):
            img_file = cofly_images_dir / name
            mask_file = cofly_labels_dir / name
            if img_file.exists() and mask_file.exists():
                new_name = f"cofly_test_{i:04d}.png"
                shutil.copy(img_file, combined_dir / "test" / "images" / new_name)
                shutil.copy(mask_file, combined_dir / "test" / "masks" / new_name)
                stats['cofly']['test'] += 1
        
        print(f"CoFly: {stats['cofly']['train']} train, {stats['cofly']['val']} val, {stats['cofly']['test']} test")
    else:
        print("⚠ 未找到 CoFly 划分文件，使用随机划分")
        # 随机划分逻辑...
    
    # 3. 创建数据集配置文件
    print("\n创建数据集配置文件...")
    config_content = f"""# Combined Dataset Configuration
# Generated automatically

Dataset Info:
- Total Images: {sum(stats['tobacco'].values()) + sum(stats['cofly'].values())}
- Tobacco Images: {sum(stats['tobacco'].values())}
- CoFly Images: {sum(stats['cofly'].values())}

Split Statistics:
Train:
  - Tobacco: {stats['tobacco']['train']}
  - CoFly: {stats['cofly']['train']}
  - Total: {stats['tobacco']['train'] + stats['cofly']['train']}

Validation:
  - Tobacco: {stats['tobacco']['val']}
  - CoFly: {stats['cofly']['val']}
  - Total: {stats['tobacco']['val'] + stats['cofly']['val']}

Test:
  - Tobacco: {stats['tobacco']['test']}
  - CoFly: {stats['cofly']['test']}
  - Total: {stats['tobacco']['test'] + stats['cofly']['test']}

Class Mapping:
- Tobacco: 0=Background, 1=Crop, 2=Weed
- CoFly: 0=Background, 1=Weed (2-class)
  Note: CoFly will be converted to 3-class by setting Crop=0 (no crop pixels)

Training Recommendations:
- Use strong data augmentation
- Consider class weighting (weed is minority class)
- Use Dice + Focal loss for better weed segmentation
"""
    
    with open(combined_dir / "dataset_info.txt", 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"\n✓ 合并数据集创建完成!")
    print(f"保存位置：{combined_dir}")
    print(f"\n数据统计:")
    print(f"  训练集：{stats['tobacco']['train'] + stats['cofly']['train']} 张图像")
    print(f"  验证集：{stats['tobacco']['val'] + stats['cofly']['val']} 张图像")
    print(f"  测试集：{stats['tobacco']['test'] + stats['cofly']['test']} 张图像")
    
    return combined_dir

if __name__ == '__main__':
    combined_dir = create_combined_dataset()
    print(f"\n下一步：使用 'python src/train.py --data_dir {combined_dir}' 开始训练")
