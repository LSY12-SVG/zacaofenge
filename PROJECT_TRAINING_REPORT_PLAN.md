# 分割项目训练汇报稿（现状解析 + 执行计划）

## 1. 项目当前状态

### 1.1 代码仓库状态
- 当前分支：`main`
- 与远端：`main...origin/main`（已同步）
- 工作区：干净（无未提交改动）

### 1.2 近期核心里程碑（最近 5 次提交）
- `64bf268`：统一训练参数接口，修复类别统计入口
- `ce9a1e0`：建立研究入口（`fpn/hfa` + DSDF + 实验脚本）
- `4da7bcb`：补齐投稿级指标定义、公平性配置、复现元信息
- `8fed6c4`：训练实时监控 + TensorBoard 接入
- `2c2aa9c`：TensorBoard 初始化容错与非 ASCII 路径回退

### 1.3 数据集训练就绪度
- 主训练数据集：`Combined_Dataset`
- 样本规模：
  - 训练集：`2150`
  - 验证集：`265`
- 全量训练集像素分布（映射后）：
  - `Background: 73.76%`
  - `Crop: 10.37%`
  - `Weed: 15.87%`
- 标签规范：训练输入统一为 `{0,1,2}`，兼容 `{0,255}` 与 `{0,128,255}` 映射

### 1.4 训练框架能力（研究版）
- 训练入口：`src/train.py`
- 支持架构：
  - `fpn | hfa | srdnet | unet | deeplabv3plus | segformerb0`
- 指标输出：
  - `mIoU`
  - `IoU(bg/crop/weed)`
  - `weed_recall`
  - `weed_tp/weed_fn/weed_fp`
  - `boundary_f1`
- 训练策略：
  - FAC（前景课程采样）
  - TSBR（边界分支）
  - 两阶段 consistency（warmup 后开启）
  - checkpoint/resume（含 optimizer/scheduler/epoch/best_iou）
- 复现元信息：
  - `params/fps/device/input_size/torch/cuda` 写入 `model_meta.yaml`

### 1.5 实验矩阵与自动化
- 配置文件：`scripts/experiments.yaml`
- 总实验数：`12`
  - `fair: 6`
  - `ablation: 6`
- 运行入口：`scripts/run_all.py`
- 汇总输出：`runs/summary_table.csv`

### 1.6 文档与监控
- 核心文档：`README.md`
- 实时终端监控：`scripts/monitor_training.py`
- Web 面板：TensorBoard（建议 `--tb_logdir C:\tb_logs\...`）

## 2. 已完成能力（汇报可强调）

- 研究入口统一：同一脚本可切换 baseline 与 proposed，减少实验口径漂移
- 公平性设计明确：`fair_*` 与 `ab_*` 分离，可回答“是否公平对比”
- 指标可追溯：有 `weed TP/FN/FP`，可解释 recall/IoU 变化来源
- 工程闭环可执行：单实验 -> 批量运行 -> 自动汇总

## 3. 当前风险与应对策略

### 3.1 训练资源与时间风险
- 风险：`12 x 60 epoch` 总耗时高
- 应对：
  - 分阶段运行（先 fair 再 ablation）
  - 每阶段结束即汇总，避免全量跑完才发现问题

### 3.2 环境一致性风险
- 风险：本机 Python 包较多，可能互相影响
- 应对：
  - 为训练建立专用虚拟环境
  - 固定关键依赖版本（torch/cuda/tensorboard/protobuf）

### 3.3 可视化面板风险
- 风险：TensorBoard 依赖异常会影响 Web 展示
- 应对：
  - 训练脚本已容错，不中断训练
  - 以 `metrics.csv + monitor_training.py` 作为保底实时监控

### 3.4 泛化结论风险
- 风险：当前主结果集中在 `Combined_Dataset`
- 应对：
  - 单列跨域测试（如 CoFly-only test）作为独立结果节

## 4. 训练完成执行计划（A-E）

### 阶段 A（1-2 天）：公平对比主表
- 目标：跑完 `fair 6`，得到 Table 1 初稿
- 命令：
```bash
python scripts/run_all.py --only fair_unet,fair_deeplabv3plus,fair_segformer_b0,fair_srdnet,fair_fpn,fair_hfa
```

### 阶段 B（2-4 天）：消融验证
- 目标：验证 FAC/TSBR/DSDF/Consistency 的增益链条
- 命令：
```bash
python scripts/run_all.py --only ab_fpn_base,ab_fpn_fac,ab_fpn_fac_tsbr,ab_hfa_logits,ab_hfa_feature,ab_hfa_full
```

### 阶段 C（1 天）：汇总与样例
- 目标：检查 `summary_table.csv`，补失败案例和可视化样本
- 产物：
  - `runs/summary_table.csv`
  - `runs/<exp>/pred_vis/*.png`

### 阶段 D（0.5-1 天）：复现确认
- 目标：冻结最优配置，固定 seed 复跑 1 次
- 验收：最优实验资产齐全（`config.yaml + metrics.csv + best.ckpt + model_meta.yaml`）

### 阶段 E（0.5 天）：汇报整理
- 建议 4 页结构：
  - 系统架构与创新点
  - 公平对比结果
  - 消融增益曲线
  - 风险与下一步

## 5. 验收标准（完成定义）

- 主表：`fair 6` 指标齐全（`mIoU/weed IoU/weed recall/boundary F1/params/fps`）
- 消融：`ablation 6` 齐全且增益链条可解释
- 复现：最优实验可完整复现（配置、权重、日志、元信息）
- 文档：命令与代码一致，避免不可运行指令

## 6. 关键入口（执行与查阅）
- `src/train.py`
- `scripts/experiments.yaml`
- `scripts/run_all.py`
- `README.md`
- `scripts/monitor_training.py`
