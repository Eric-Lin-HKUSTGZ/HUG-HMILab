# HUG-HMI-Lab

HUG（Human Universal Grasping）在 HMILab 手物交互项目中的可运行集成版本。
本仓库包含 HUG 原始推理代码，以及训练、数据准备和可视化工具。

给定一张 RGB-D 图像、相机内参和物体上的查询点，HUG 生成 99D MANO 抓取状态，并解码为手部关键点、网格和手腕位姿。

```text
RGB-D + query point
    ├─ DINOv2（冻结，RGB语义）
    ├─ PointNeXt（可训练，点云几何）
    ├─ PointPainting + Transformer（多模态融合）
    └─ Flow Matching DiT（50步Euler采样）
             ↓
       MANO 右手模型
             ↓
  landmarks / mesh / grasp.pkl
```

## 目录

- `src/app.py`：Viser 交互式应用，点击图像生成抓取。
- `src/inference.py`：批量推理并写出 `grasp_pred/*.pkl`。
- `src/visualize_predictions.py`：离线预测可视化 Web 服务。
- `src/prepare_inputs.py`：任意 RGB-D + 内参转换为统一 224×224 pkl。
- `src/train.py`：DDP 训练入口（1M-HUGs 抓取生成数据集）。
- `src/models/`：DINOv2、PointNeXt、PointPainting、DiT、MANO。
- `scripts/`：DINOv2 权重转换、1M-HUGs 验证集划分、预测渲染脚本。
- `analysis_docs/`：模型、数据和推理流程说明。
- `assets/`：MANO 资产、mesh faces、shape 和归一化统计量。

## 环境与权重

服务器上的验证环境：Python 3.10、PyTorch 2.9.1、CUDA 12.8，conda 环境为 `hug`。
推荐直接使用该环境：

```bash
cd /root/code/HUG-HMILab
/root/code/vepfs/miniconda3/envs/hug/bin/pip install -e .
```

本仓库使用的权重放在共享目录，不重复提交到 Git：

```bash
CKPT=/root/code/vepfs/HUG-for-Recon-Gen/hug_checkpoint/hug_full.safetensors
DINOV2=/root/code/vepfs/HUG-for-Recon-Gen/dinov2
```

- HUG checkpoint：`$CKPT`
- DINOv2 原始/转换权重：`$DINOV2`
- MANO 模型：已复制到 `assets/mano/`（右手模型位于 `assets/mano/models/MANO_RIGHT.pkl`）。

如果在新机器上部署，请准备 MANO 官方模型，并确保 `assets/mano/` 与 `src/utils/data_keys.py` 的路径一致。

## 推理

### HUG-Bench 或已有 pkl

```bash
cd /root/code/HUG-HMILab
/root/code/vepfs/miniconda3/envs/hug/bin/python -m hug.inference \
  --checkpoint-path "$CKPT" \
  --dataset-path data/hug_bench \
  --num-samples 1 \
  --sampling-steps 50 \
  --batch-size 1
```

预测会写入 `data/hug_bench/grasp_pred/`。

### 自定义 RGB-D 输入

输入目录需要包含：

- `rgb.png` 或 `rgb.jpg`：8-bit RGB 图像；
- `depth.png`：uint16 单通道深度，单位为毫米，且与 RGB 对齐；
- `intrinsics.txt`、`.npy` 或 `.json`：四元组 `fx fy cx cy` 或 3×3 K 矩阵。

```bash
/root/code/vepfs/miniconda3/envs/hug/bin/python -m hug.prepare_inputs \
  --dataset-path data/custom

/root/code/vepfs/miniconda3/envs/hug/bin/python -m hug.inference \
  --checkpoint-path "$CKPT" \
  --dataset-path data/custom \
  --sample-name custom \
  --sampling-steps 50 \
  --batch-size 1
```

`prepare_inputs` 会将图像中心作为无 mask 自定义输入的默认 query，并生成 `custom.pkl`；推理结果为 `data/custom/grasp_pred/custom.pkl`。

### 交互式点击应用

```bash
/root/code/vepfs/miniconda3/envs/hug/bin/python -m hug.app \
  --checkpoint-path "$CKPT" \
  --dataset-path data/hug_bench \
  --port 8080 \
  --save-pred
```

打开 `http://localhost:8080`（远程服务器使用端口转发），在左侧图像点击物体即可生成抓取；每次点击可选保存到 `grasp_pred/`。

## 预测可视化

```bash
/root/code/vepfs/miniconda3/envs/hug/bin/python -m hug.visualize_predictions \
  --dataset-path data/custom \
  --sample-name custom \
  --port 18080
```

服务启动后访问 `http://localhost:18080`。该命令会持续运行，按 `Ctrl+C` 停止。

### 生成静态 PNG

如果需要直接查看或批量归档图片，可使用纯 Python 渲染脚本：

```bash
/root/code/vepfs/miniconda3/envs/hug/bin/python scripts/render_predictions.py \
  --dataset-path data/hug_bench \
  --output-dir prediction_images
```

每个样本会生成一张六联图（RGB+2D joints、depth、front/side/angled 3D hand views 和信息面板），输出在 `prediction_images/`。

## 训练

训练配置为 `configs/train_hug.yaml`，数据为 HUG 官方 1M-HUGs 抓取生成数据集
（`/root/code/vepfs/dataset/1m-hugs/grasp_data`，约 1.28M 帧；验证集为
录制级留出 split，见 `scripts/README.md`）。

```bash
torchrun --nproc_per_node=4 -m hug.train \
  --config configs/train_hug.yaml
```

冒烟测试（少量步数 + 截断训练集，走完整 train/val/checkpoint 流程）：

```bash
torchrun --nproc_per_node=4 -m hug.train \
  --config configs/train_hug.yaml \
  --max-steps 30 --max-train-samples 20000
```

要点：

- 输出目录与训练日志写在 vepfs（`output_dir`），勿写根分区（仅 20G）；
  checkpoint 保存时剥离冻结 DINOv2（单文件 ~0.9G）。
- 验证走真实推理路径 `sample()`（50 步 ODE 采样）不算 loss，指标为
  MPJPE / PA-MPJPE / MPVPE / PA-MPVPE（mm）；多卡分片评估后 all_reduce 聚合。
- `model_best.pt` 按验证 score 自动保存；`model.pt` 为周期性续训点，
  checkpoint 内嵌 cfg + norm_stats，可直接被 `inference.py` / `app.py` 加载。
- 纯 PyTorch PointNeXt 会在 SA 层构造较大的 kNN 临时张量；训练时应使用多卡
  和合适的 batch size。显存有限时可先降低 batch size 或用单卡 smoke test；
  这不影响单样本推理和可视化。
- `output_dir/logs/` 下每个 rank 有完整文本日志与 error 日志；rank 0 另有
  结构化 JSONL 训练日志（含 startup/train/validation/checkpoint 事件）。

## 数据约定

- RGB 和深度会统一裁剪/缩放到 224×224，内参 K 同步调整。
- 深度反投影为相机坐标系米制点云，默认以 query 点为中心做 0.2～0.3 m 球裁剪，再采样 4096 点。
- MANO 状态为 `t(3) + wrist_R6D(6) + finger_pose_6D(90)`。
- 训练时 query 点从 mask 内随机采样；推理 pkl 使用其 `condition_point`。
- 推理 checkpoint 自带模型配置和 norm stats，避免模态开关与权重不匹配。

## 论文

```bibtex
@article{wu2026hug,
  title={Human Universal Grasping},
  author={Kevin Yuanbo Wu and Tianxing Zhou and Isaac Tu and Billy Yan and Irmak Guzey and David Fouhey and Dandan Shan and Lerrel Pinto},
  journal={arXiv preprint arXiv:2606.17054},
  year={2026}
}
```
