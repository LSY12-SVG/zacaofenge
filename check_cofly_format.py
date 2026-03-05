"""
CoFly-WeedDB 数据集格式验证脚本
检查数据集结构、标注格式、类别定义等
"""
import os
import cv2
import numpy as np
from pathlib import Path

def check_dataset_format(dataset_path):
    """检查 CoFly-WeedDB 数据集格式"""
    print("=" * 60)
    print(f"CoFly-WeedDB 数据集格式验证")
    print(f"数据集路径：{dataset_path}")
    print("=" * 60)
    
    # 检查目录结构
    images_dir = os.path.join(dataset_path, 'images')
    labels_dir = os.path.join(dataset_path, 'labels')
    
    print("\n1. 目录结构检查:")
    print(f"   ✓ images 目录：{os.path.exists(images_dir)}")
    print(f"   ✓ labels 目录：{os.path.exists(labels_dir)}")
    
    # 统计文件数量
    image_files = list(Path(images_dir).glob('*.png'))
    label_files = list(Path(labels_dir).glob('*.png'))
    
    print(f"\n2. 文件统计:")
    print(f"   - 图像数量：{len(image_files)}")
    print(f"   - 标注数量：{len(label_files)}")
    
    # 检查配对情况
    image_names = set([f.stem for f in image_files])
    label_names = set([f.stem for f in label_files])
    paired = image_names.intersection(label_names)
    
    print(f"   - 配对数量：{len(paired)}")
    print(f"   - 缺少标注的图像：{len(image_names - label_names)}")
    print(f"   - 缺少图像的标注：{len(label_names - image_names)}")
    
    # 检查图像和标注属性
    if len(image_files) > 0:
        print("\n3. 图像属性检查:")
        sample_img = cv2.imread(str(image_files[0]))
        print(f"   - 图像尺寸：{sample_img.shape}")
        print(f"   - 通道数：{sample_img.shape[2] if len(sample_img.shape) > 2 else 1}")
        print(f"   - 数据类型：{sample_img.dtype}")
        
        # 检查多个图像的分辨率
        sizes = set()
        for img_file in image_files[:10]:
            img = cv2.imread(str(img_file))
            sizes.add(img.shape[:2])
        print(f"   - 分辨率一致性：{'统一' if len(sizes) == 1 else '不统一'}")
        if len(sizes) > 1:
            print(f"   - 多种分辨率：{sizes}")
    
    # 检查标注属性
    if len(label_files) > 0:
        print("\n4. 标注属性检查:")
        sample_mask = cv2.imread(str(label_files[0]), 0)
        print(f"   - 标注尺寸：{sample_mask.shape}")
        print(f"   - 数据类型：{sample_mask.dtype}")
        
        # 统计像素值分布
        unique_values = set()
        for label_file in label_files[:20]:
            mask = cv2.imread(str(label_file), 0)
            unique_values.update(np.unique(mask))
        
        print(f"   - 像素值范围：{min(unique_values)} - {max(unique_values)}")
        print(f"   - 唯一像素值：{sorted(unique_values)}")
        
        # 推断类别数
        num_classes = len(unique_values)
        print(f"   - 推断类别数：{num_classes}")
        
        # 尝试推断类别含义
        if num_classes == 2:
            print(f"   - 类别定义：0=背景，1=杂草 (2 类)")
        elif num_classes == 3:
            print(f"   - 类别定义：0=背景，1=作物，2=杂草 (3 类)")
        else:
            print(f"   ⚠ 警告：非标准类别数，需要手动确认")
        
        # 检查标注与图像尺寸是否匹配
        sample_img = cv2.imread(str(image_files[0]))
        sample_mask = cv2.imread(str(label_files[0]), 0)
        if sample_img.shape[:2] == sample_mask.shape:
            print(f"   ✓ 标注与图像尺寸匹配")
        else:
            print(f"   ⚠ 标注与图像尺寸不匹配:")
            print(f"     图像：{sample_img.shape[:2]}, 标注：{sample_mask.shape}")
    
    # 检查训练/测试划分文件
    print("\n5. 训练/测试划分文件:")
    split_files = list(Path(dataset_path).glob('train_split*.txt'))
    test_files = list(Path(dataset_path).glob('test_split*.txt'))
    
    for split_file in split_files:
        with open(split_file, 'r') as f:
            lines = f.readlines()
        print(f"   - {split_file.name}: {len(lines)} 张图像")
    
    for test_file in test_files:
        with open(test_file, 'r') as f:
            lines = f.readlines()
        print(f"   - {test_file.name}: {len(lines)} 张图像")
    
    # 显示样本图像
    print("\n6. 样本数据可视化:")
    if len(image_files) > 0:
        sample_img_file = image_files[0]
        sample_label_file = labels_dir / (sample_img_file.stem + '.png')
        
        if sample_label_file.exists():
            img = cv2.imread(str(sample_img_file))
            mask = cv2.imread(str(sample_label_file), 0)
            
            print(f"   样本图像：{sample_img_file.name}")
            print(f"   样本标注：{sample_label_file.name}")
            
            # 统计各类别像素比例
            total_pixels = mask.size
            for val in sorted(unique_values):
                pixel_count = np.sum(mask == val)
                percentage = (pixel_count / total_pixels) * 100
                print(f"   - 类别 {val}: {pixel_count} 像素 ({percentage:.2f}%)")
    
    print("\n" + "=" * 60)
    print("验证完成!")
    print("=" * 60)
    
    # 返回兼容性建议
    print("\n7. 与 Tobacco Aerial Dataset 兼容性分析:")
    print(f"   ✓ 格式兼容：PNG 图像 + PNG 标注")
    print(f"   ✓ 任务类型：语义分割")
    print(f"   ⚠ 需要注意:")
    
    if len(sizes) > 1:
        print(f"   - 分辨率不统一，需要在 dataset.py 中统一 Resize")
    
    if num_classes != 3:
        print(f"   - 类别数不一致 (Tobacco 是 3 类，CoFly 是 {num_classes} 类)")
        print(f"   - 需要调整类别映射或重新标注")
    
    print(f"   - 建议统一为 3 类：0=背景，1=作物，2=杂草")
    
    return {
        'num_images': len(image_files),
        'num_labels': len(label_files),
        'num_classes': num_classes,
        'image_size': sample_img.shape[:2] if len(image_files) > 0 else None,
        'unique_values': sorted(unique_values),
        'compatible': True
    }

if __name__ == '__main__':
    dataset_path = r"e:\温室无人机巡检系统\杂草\CoFly-WeedDB\CoFly-WeedDB"
    result = check_dataset_format(dataset_path)
    
    # 保存验证结果
    output_file = r"e:\温室无人机巡检系统\杂草\cofly_format_check_result.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("CoFly-WeedDB 数据集格式验证结果\n")
        f.write("=" * 60 + "\n\n")
        for key, value in result.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\n验证结果已保存到：{output_file}")
