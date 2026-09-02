# HUG-HMI-Lab

HUG（Human Universal Grasping）在 HMILab 手物交互项目中的可运行集成版本。
本仓库包含 HUG 原始推理代码，以及从 `HUG-for-Recon-Gen` 迁移的训练、评测、数据转换和可视化工具。

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
- `src/train.py`：DDP 训练入口，支持 DexYCB/HO3D 混合训练。
- `src/eval_test.py`：官方测试集评估（MPJPE、PA-MPJPE、MPVPE、PA-MPVPE）。
- `src/models/`：DINOv2、PointNeXt、PointPainting、DiT、MANO。
- `scripts/`：HO3D/DexYCB 转换、划分、统计和数据检查脚本。
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

如果需要直接查看或批量归档图片，可使用迁移的纯 Python 渲染脚本：

```bash
/root/code/vepfs/miniconda3/envs/hug/bin/python scripts/render_predictions.py \
  --dataset-path data/hug_bench \
  --output-dir prediction_images
```

每个样本会生成一张六联图（RGB+2D joints、depth、front/side/angled 3D hand views 和信息面板），输出在 `prediction_images/`。

## 训练与评测

训练配置：

- `configs/train_hug.yaml`：HUG 原始 1M-HUGs 训练配置；
- `configs/train_handrecon.yaml`：DexYCB + HO3D 微调配置。

完整训练示例：

```bash
torchrun --nproc_per_node=4 -m hug.train \
  --config configs/train_handrecon.yaml
```

官方测试集评估：

```bash
torchrun --nproc_per_node=4 -m hug.eval_test \
  --config configs/train_handrecon.yaml
```

纯 PyTorch PointNeXt 会在 SA 层构造较大的 kNN 临时张量；训练时应使用多卡和合适的 batch size。显存有限时可先降低 batch size 或使用单卡 smoke test；这不影响单样本推理和可视化。

## 数据约定

- RGB 和深度会统一裁剪/缩放到 224×224，内参 K 同步调整。
- 深度反投影为相机坐标系米制点云，默认以 query 点为中心做 0.2～0.3 m 球裁剪，再采样 4096 点。
- MANO 状态为 `t(3) + wrist_R6D(6) + finger_pose_6D(90)`。
- 训练时 query 点从 mask 内随机采样；评测 pkl 使用其 `condition_point`。
- 推理 checkpoint 自带模型配置和 norm stats，避免模态开关与权重不匹配。

HO3D/DexYCB 转换、6D 旋转约定、MANO wrist 平移参考点和评测口径的修复说明见 `analysis_docs/handrecon_conversion_bugs.md`。

## 论文

```bibtex
@article{wu2026hug,
  title={Human Universal Grasping},
  author={Kevin Yuanbo Wu and Tianxing Zhou and Isaac Tu and Billy Yan and Irmak Guzey and David Fouhey and Dandan Shan and Lerrel Pinto},
  journal={arXiv preprint arXiv:2606.17054},
  year={2026}
}
```
