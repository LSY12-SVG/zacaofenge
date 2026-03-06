"""检查 CoFly 数据集的标注像素值"""
import cv2
import numpy as np
from pathlib import Path

# 读取几个标注文件检查像素值
labels_dir = Path(r"e:\温室无人机巡检系统\杂草\CoFly-WeedDB\CoFly-WeedDB\labels")
label_files = list(labels_dir.glob("*.png"))

print(f"找到 {len(label_files)} 个标注文件\n")

# 检查前 10 个文件
unique_values_all = set()
for i, label_file in enumerate(label_files[:10]):
    mask = cv2.imread(str(label_file), 0)
    unique_vals = np.unique(mask)
    unique_values_all.update(unique_vals)
    print(f"{i+1}. {label_file.name}")
    print(f"   尺寸：{mask.shape}, 像素值：{unique_vals}")

print(f"\n所有唯一像素值：{sorted(unique_values_all)}")
print(f"类别数：{len(unique_values_all)}")

if len(unique_values_all) == 2:
    print("✓ 2 类分割：0=背景，1=杂草")
elif len(unique_values_all) == 3:
    print("✓ 3 类分割：0=背景，1=作物，2=杂草")
else:
    print(f"⚠ 非标准类别数：{len(unique_values_all)}")
