# 温室无人机巡检系统 - 杂草分割项目结构

## 📁 项目总览

```
杂草/
├── 📦 核心代码 (src/)
├── 📚 数据集 (Combined_Dataset/, CoFly-WeedDB/)
├── 🤖 模型架构 (models/)
├── 🧪 测试脚本
├── 📖 文档
└── 🛠️ 配置文件
```

---

## 1️⃣ 核心代码部分 (`src/`)

### 1.1 训练相关
| 文件 | 功能 | 说明 |
|------|------|------|
| [`train.py`](src/train.py) | **主训练脚本** | 支持多种模型（U-Net, DeepLabV3+, MAnet, **SRDNet**） |
| [`evaluate.py`](src/evaluate.py) | 模型评估 | 在测试集上计算 IoU、mIoU 等指标 |
| [`predict.py`](src/predict.py) | 预测推理 | 单张图像或批量预测 |
| [`run_paper_experiments.py`](src/run_paper_experiments.py) | 论文实验 | 运行对比实验和消融实验 |

### 1.2 数据相关
| 文件 | 功能 | 说明 |
|------|------|------|
| [`dataset.py`](src/dataset.py) | **数据集加载器** | 支持 Tobacco、CoFly、合并数据集 |
| [`merge_datasets.py`](merge_datasets.py) | 数据集合并 | 合并 Tobacco 和 CoFly 数据集 |
| [`check_cofly_format.py`](check_cofly_format.py) | 数据检查 | 检查 CoFly 数据集格式 |
| [`check_cofly_labels.py`](check_cofly_labels.py) | 标签检查 | 检查 CoFly 标签分布 |

### 1.3 可视化相关
| 文件 | 功能 | 说明 |
|------|------|------|
| [`visualize.py`](src/visualize.py) | 结果可视化 | 可视化预测结果和对比 |
| [`draw_architecture.py`](src/draw_architecture.py) | 架构图绘制 | 绘制模型架构图 |
| [`draw_architecture_mpl.py`](src/draw_architecture_mpl.py) | 架构图 (Matplotlib) | 使用 matplotlib 绘制 |

### 1.4 模型定义
| 文件 | 功能 | 说明 |
|------|------|------|
| [`model.py`](src/model.py) | **基础 U-Net** | 自定义 U-Net 实现 |
| [`model_advanced.py`](src/model_advanced.py) | **高级模型工厂** | DeepLabV3+, U-Net++, MAnet, Linknet, PSPNet |
| [`model_transformer.py`](src/model_transformer.py) | Transformer 模型 | SegFormer 等 Vision Transformer |

---

## 2️⃣ SRDNet 模型架构 (`src/models/`)

### 2.1 SRDNet 核心模块

```
src/models/
├── __init__.py                      # 包初始化
├── srdnet.py                        # SRDNet 主模型
├── crop_structure_head.py           # 作物结构先验建模
├── residual_extractor.py            # 残差特征提取器
├── frequency_enhancement.py         # 频域增强模块
└── decoder.py                       # 轻量级解码器
```

| 模块 | 功能 | 输入 → 输出 | 核心创新 |
|------|------|-----------|---------|
| [`srdnet.py`](src/models/srdnet.py) | **SRDNet 主模型** | Image → Segmentation Mask | 整体架构整合 |
| [`crop_structure_head.py`](src/models/crop_structure_head.py) | 作物结构预测 | Feature Map → Crop Probability Map | 农业领域 Inductive Bias |
| [`residual_extractor.py`](src/models/residual_extractor.py) | 残差解耦 | F_b - α·F_c → Weed-enhanced Feature | 语义层面"去作物化" |
| [`frequency_enhancement.py`](src/models/frequency_enhancement.py) | 频域增强 | Residual Feature → Hybrid Feature | Laplacian/FFT高频增强 |
| [`decoder.py`](src/models/decoder.py) | 轻量解码器 | Feature → Logits | CBAM 注意力 + 深度可分离卷积 |

### 2.2 SRDNet 整体架构流程

```
Input Image (3, H, W)
         ↓
Backbone (ConvNeXt-Tiny) → F_b (768, H/32, W/32)
         ↓
Stage1: Crop Structure Head → M_c (作物概率图)
         ↓
Stage2: Residual Feature Extractor → F_r (残差特征)
         ↓
Stage3: Frequency Enhancement Block → F_hybrid (混合特征)
         ↓
Stage4: Lightweight Decoder → Logits (n_classes, H, W)
         ↓
Segmentation Output (上采样到原图尺寸)
```

---

## 3️⃣ 损失函数 (`src/losses/`)

### 3.1 损失函数模块

```
src/losses/
├── __init__.py                      # 包初始化
└── boundary_loss.py                 # 边界损失

src/loss.py                          # 主损失函数文件
```

| 损失函数 | 文件 | 功能 | 适用场景 |
|---------|------|------|---------|
| [`DiceLoss`](src/loss.py) | loss.py | Dice 系数损失 | 类别不平衡 |
| [`FocalLoss`](src/loss.py) | loss.py | Focal 损失 | 难分样本挖掘 |
| [`DiceCELoss`](src/loss.py) | loss.py | Dice + CE | 通用分割 |
| [`DiceFocalLoss`](src/loss.py) | loss.py | Dice + Focal | 杂草分割推荐 |
| [`SRDNetLoss`](src/loss.py) | loss.py | **Dice + Focal + Boundary** | SRDNet 专用 |
| [`BoundaryLoss`](src/losses/boundary_loss.py) | boundary_loss.py | 边界损失 (Sobel) | 边缘增强 |

---

## 4️⃣ 数据集部分

### 4.1 合并数据集 (`Combined_Dataset/`)

```
Combined_Dataset/
├── train/
│   ├── images/          # 训练图像 (2150 张)
│   └── masks/           # 训练标签
├── val/
│   ├── images/          # 验证图像 (265 张)
│   └── masks/           # 验证标签
├── test/
│   ├── images/          # 测试图像 (306 张)
│   └── masks/           # 测试标签
└── dataset_info.txt     # 数据集说明
```

**数据集统计**:
- **训练集**: 2150 张 (Tobacco: 2016, CoFly: 134)
- **验证集**: 265 张 (Tobacco: 252, CoFly: 13)
- **测试集**: 306 张 (Tobacco: 252, CoFly: 54)

**类别映射**:
- **Tobacco**: 0=背景，1=作物，2=杂草
- **CoFly**: 0=背景，1=杂草 → 转换为 0=背景，2=杂草

### 4.2 CoFly-WeedDB 原始数据集 (`CoFly-WeedDB/`)

```
CoFly-WeedDB/
└── CoFly-WeedDB/
    ├── images/              # 原始图像
    ├── labels/              # 原始标签
    ├── train_split1.txt     # 训练集划分 (split 1)
    ├── train_split2.txt     # 训练集划分 (split 2)
    ├── train_split3.txt     # 训练集划分 (split 3)
    ├── val_split1.txt       # 验证集划分
    ├── test_split1.txt      # 测试集划分
    └── ...
```

### 4.3 数据集压缩包
- [`CoFly-WeedDB.zip`](CoFly-WeedDB.zip) - CoFly 数据集原始压缩包

---

## 5️⃣ 测试脚本

| 文件 | 功能 | 状态 |
|------|------|------|
| [`test_srdnet.py`](test_srdnet.py) | SRDNet 完整测试 | 测试前向传播、作物 mask 提取 |
| [`test_srdnet_simple.py`](test_srdnet_simple.py) | SRDNet 简化测试 | 快速测试导入和创建 |
| [`test_import.py`](test_import.py) | 导入测试 | 测试各模块导入 |
| [`test_cpu.py`](test_cpu.py) | CPU 模式测试 | 禁用 CUDA 测试 |
| [`debug_import.py`](debug_import.py) | 导入调试 | 详细记录导入时间 |
| [`debug_train.py`](debug_train.py) | 训练调试 | 分步诊断训练流程 |
| [`quick_test.py`](quick_test.py) | 快速测试 | 简化版 SRDNet 测试 |

---

## 6️⃣ 文档

| 文件 | 类型 | 内容 |
|------|------|------|
| [`SRDNet_README.md`](SRDNet_README.md) | **使用指南** | SRDNet 完整使用说明 |
| [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md) | 训练指南 | 合并数据集训练教程 |
| [`.trae/documents/SRDNet_Implementation_Plan.md`](.trae/documents/SRDNet_Implementation_Plan.md) | **实现计划** | SRDNet 详细设计和实现步骤 |

---

## 7️⃣ 配置文件

| 文件 | 功能 |
|------|------|
| [`requirements.txt`](requirements.txt) | Python 依赖包列表 |
| [`.venv/`](.venv/) | Python 虚拟环境 (不建议修改) |

---

## 8️⃣ 辅助工具

### 8.1 数据预处理
- [`merge_datasets.py`](merge_datasets.py) - 合并 Tobacco 和 CoFly 数据集
- [`check_cofly_format.py`](check_cofly_format.py) - 检查 CoFly 数据格式
- [`check_cofly_labels.py`](check_cofly_labels.py) - 分析 CoFly 标签分布

### 8.2 可视化
- [`visualize.py`](src/visualize.py) - 预测结果可视化
- [`draw_architecture.py`](src/draw_architecture.py) - 模型架构图绘制
- [`draw_architecture_mpl.py`](src/draw_architecture_mpl.py) - Matplotlib 版本

---

## 9️⃣ 使用流程

### 9.1 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 测试 SRDNet 模型
python test_srdnet_simple.py

# 3. 开始训练
python src/train.py \
    --data_dir "Combined_Dataset" \
    --model srdnet \
    --backbone convnext_tiny \
    --epochs 60 \
    --batch_size 4 \
    --lr 5e-5 \
    --loss srdnet_loss \
    --save_name srdnet_combined.pth
```

### 9.2 评估模型

```bash
# 在测试集上评估
python src/evaluate.py \
    --data_dir "Combined_Dataset" \
    --model_path "srdnet_combined.pth" \
    --model srdnet
```

### 9.3 可视化预测

```bash
# 预测单张图像
python src/predict.py \
    --image_path "path/to/image.jpg" \
    --model_path "srdnet_combined.pth" \
    --model srdnet \
    --save_path "predictions/"
```

---

## 🔟 项目结构总结

```
杂草/
│
├── 📦 核心代码 (src/)
│   ├── train.py, evaluate.py, predict.py        # 训练/评估/预测
│   ├── dataset.py                               # 数据集加载
│   ├── loss.py                                  # 损失函数
│   ├── model.py, model_advanced.py              # 模型定义
│   │
│   ├── models/                                  # SRDNet 模块
│   │   ├── srdnet.py                            # 主模型
│   │   ├── crop_structure_head.py               # 作物结构头
│   │   ├── residual_extractor.py                # 残差提取器
│   │   ├── frequency_enhancement.py             # 频域增强
│   │   ├── decoder.py                           # 解码器
│   │   └── __init__.py
│   │
│   └── losses/                                  # 损失函数模块
│       ├── boundary_loss.py                     # 边界损失
│       └── __init__.py
│
├── 📚 数据集
│   ├── Combined_Dataset/                        # 合并数据集
│   │   ├── train/, val/, test/
│   │   └── dataset_info.txt
│   ├── CoFly-WeedDB/                            # CoFly 原始数据集
│   └── CoFly-WeedDB.zip
│
├── 🧪 测试脚本
│   ├── test_srdnet.py, test_srdnet_simple.py    # SRDNet 测试
│   ├── test_import.py, test_cpu.py              # 导入测试
│   └── debug_import.py, debug_train.py          # 调试脚本
│
├── 📖 文档
│   ├── SRDNet_README.md                         # SRDNet 使用指南
│   ├── TRAINING_GUIDE.md                        # 训练指南
│   └── .trae/documents/SRDNet_Implementation_Plan.md
│
├── 🛠️ 配置
│   ├── requirements.txt                         # 依赖
│   └── .venv/                                   # 虚拟环境
│
└── 🔧 辅助工具
    ├── merge_datasets.py                        # 数据合并
    ├── check_cofly_*.py                         # 数据检查
    └── visualize.py, draw_architecture.py       # 可视化
```

---

## 📊 关键文件索引

### 训练相关
- **主训练脚本**: [`src/train.py`](src/train.py)
- **损失函数**: [`src/loss.py`](src/loss.py)
- **数据集**: [`src/dataset.py`](src/dataset.py)

### 模型相关
- **基础模型**: [`src/model.py`](src/model.py), [`src/model_advanced.py`](src/model_advanced.py)
- **SRDNet**: [`src/models/srdnet.py`](src/models/srdnet.py)
- **SRDNet 组件**: 
  - [`crop_structure_head.py`](src/models/crop_structure_head.py)
  - [`residual_extractor.py`](src/models/residual_extractor.py)
  - [`frequency_enhancement.py`](src/models/frequency_enhancement.py)
  - [`decoder.py`](src/models/decoder.py)

### 文档相关
- **使用指南**: [`SRDNet_README.md`](SRDNet_README.md)
- **实现计划**: [`.trae/documents/SRDNet_Implementation_Plan.md`](.trae/documents/SRDNet_Implementation_Plan.md)

---

## 🎯 下一步建议

1. **解决导入慢的问题**: 
   - 检查 CUDA 配置
   - 考虑使用 CPU 模式训练（较慢但稳定）
   
2. **开始训练**:
   - 使用简化配置先测试：`--epochs 5 --batch_size 2`
   - 确认模型可以正常训练后再全量训练

3. **性能对比**:
   - 运行现有模型（DeepLabV3+, MAnet）作为基线
   - 训练 SRDNet 并对比性能

---

**最后更新**: 2026-03-04  
**项目状态**: SRDNet 已实现完成，等待训练验证
