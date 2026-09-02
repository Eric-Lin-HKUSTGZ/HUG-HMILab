# HUG 手部重建训练任务说明

基于 HUG 框架的手部重建（hand reconstruction）训练任务：输入 RGB-D + 点云，
用 flow transformer（DiT）回归 99D MANO 抓取状态（`t(3) + R_6d(6) + pose_6d(15×6)`），
数据为 **HO3D_v3 + DexYCB 混合训练**。

## 1. 训练指令

```bash
cd /root/code/HUG-for-Recon-Gen
torchrun --nproc_per_node=4 -m src.train --config configs/train_handrecon.yaml
```

- 配置文件：`configs/train_handrecon.yaml`（本文件第 3 节有摘要）
- `/root/code/vepfs/HUG-for-Recon-Gen/hand_recon/`（`output_dir`，vepfs 大文件系统）与
  其下的 `train_log.jsonl`（`train.log_file` 可配置路径+文件名；相对路径按
  `output_dir` 解析，默认 `<output_dir>/train_log.jsonl`）

### 冒烟测试（smoke test）

```bash
torchrun --nproc_per_node=4 -m src.train --config configs/train_handrecon.yaml \
    --max-steps 10 --max-train-samples 20000
```

⚠️ 冒烟测试会覆盖 `output_dir` 里的日志，正式训练前建议把 smoke 的
`output_dir` 改成别的目录（如 `outputs/hand_recon_smoke`），或在正式训练前清空。

## 2. 数据

### 数据来源（转换产物，由 `GraspDataset` 直接消费）

| 数据集 | 路径 | train 帧数 | val 帧数 | test 帧数（官方） |
|---|---|---|---|---|
| HO3D_v3 | `/root/code/vepfs/dataset/hand_recon_hug/ho3d` | 77,019 | 5,760 | 17,224（evaluation split） |
| DexYCB | `/root/code/vepfs/dataset/hand_recon_hug/dexycb_v2_canonical_right` | 394,193 | 21,785 | 76,845（s0_test） |
| 合计 | — | **471,402** | 27,546 | **94,069** |

划分清单（stem 列表，位于数据目录外的 `splits/`）：
`/root/code/vepfs/dataset/hand_recon_hug/splits_v2/` 下
`{ho3d,dexycb}_{train,val,test}.clean.txt` 与 `ho3d_eval.clean.txt`（canonical v2）

- HO3D train/val：按序列（recording-level）留出验证集；test = 官方
  evaluation split（无 MANO，只有 joints/verts GT）
- DexYCB：官方 s0 split（s0_train / s0_val / s0_test）。canonical v2
  **全量转换**（508,384 pkl 覆盖 train/val/test 全部有标注帧），并依据
  `meta.yml["mano_sides"][0]` 处理左右手；左手镜像到右手 canonical 空间。
  test 清单按 `s0_test.jsonl` 检索生成，不是重新划分
- `.clean` 过滤规则（`filter_empty_masks.py`，`--lists` 可指定子集）：
  HO3D train / DexYCB 全部 = 手部 mask 非空（剔 ~3%）；HO3D eval =
  手腕投影（`condition_point`）在 224 画面内（官方 eval 集不发布分割
  mask，`object_mask` 为空字节，剔 14.5%）。想严格按官方全集评测可改用
  `.txt` 清单

归一化统计：`assets/norm_stats_handrecon_v2.json`（仅用 canonical v2 训练集计算，无 test 泄漏）。

### 当前 canonical v2 状态

- DexYCB canonical 数据：`/root/code/vepfs/dataset/hand_recon_hug/dexycb_v2_canonical_right`
  （508,384 pkl；left/right 统一为 right canonical）
- 官方清单不变，仅切换到 v2 pkl 目录后重新生成 clean 列表：train 394,193、
  val 21,785、test 76,845；HO3D eval 17,224
- norm stats：`assets/norm_stats_handrecon_v2.json`，训练样本 n=477,518
- 2000 样本 parity audit：left 3D mean 7.2mm / P90 12.3mm，right mean 2.8mm /
  P90 5.0mm；2D internal max=0px
- 当前配置 `configs/train_handrecon.yaml` 已指向 v2 数据、`splits_v2` 和 v2 stats；
  smoke training 已通过。旧 DexYCB pkl 和旧 checkpoint 不要混用

### 当前 canonical v2 状态

- DexYCB canonical 数据：`/root/code/vepfs/dataset/hand_recon_hug/dexycb_v2_canonical_right`
  （508,384 pkl；left/right 统一为 right canonical）
- 官方划分不变，只切换到 v2 pkl 目录后生成对应 clean 清单：DexYCB train
  394,193、val 21,785、test 76,845；HO3D eval 17,224
- norm stats：`assets/norm_stats_handrecon_v2.json`，仅使用训练清单，n=477,518
- 2000 样本 parity audit：left 3D mean 7.2mm / P90 12.3mm，right mean 2.8mm /
  P90 5.0mm；2D internal max=0px
- 当前配置 `configs/train_handrecon.yaml` 已指向 v2 数据、`splits_v2` 和 v2 stats；
  2 卡训练 smoke 已通过。旧 DexYCB pkl 和旧 checkpoint 不要混用

### 数据准备 pipeline（已跑完，仅备忘）

```bash
# 1. 原始数据 -> HUG pkl schema（详见 scripts/README.md）
python scripts/convert_ho3d.py --out-dir .../hand_recon_hug/ho3d
python scripts/convert_dexycb.py \
    --out-dir .../hand_recon_hug/dexycb_v2_canonical_right

# 2. 官方 train/val/test stem 清单（划分不变；DexYCB 显式使用 canonical v2）
python scripts/make_handrecon_splits.py \
    --root /root/code/vepfs/dataset/hand_recon_hug \
    --dexycb-dir /root/code/vepfs/dataset/hand_recon_hug/dexycb_v2_canonical_right \
    --split-dir /root/code/vepfs/dataset/hand_recon_hug/splits_v2 \
    --ho3d-val-sequences 5

# 2b. clean 清单（DexYCB 显式指定 v2 目录）
python scripts/filter_empty_masks.py \
    --root /root/code/vepfs/dataset/hand_recon_hug \
    --dataset-path /root/code/vepfs/dataset/hand_recon_hug/dexycb_v2_canonical_right \
    --split-dir /root/code/vepfs/dataset/hand_recon_hug/splits_v2 \
    --lists dexycb_train,dexycb_val,dexycb_test

# 3. 计算归一化统计（仅训练清单；不读 val/test）
python scripts/compute_norm_stats.py \
    --data-dirs /root/code/vepfs/dataset/hand_recon_hug/ho3d \
                /root/code/vepfs/dataset/hand_recon_hug/dexycb_v2_canonical_right \
    --split-files /root/code/vepfs/dataset/hand_recon_hug/splits_v2/ho3d_train.clean.txt \
                  /root/code/vepfs/dataset/hand_recon_hug/splits_v2/dexycb_train.clean.txt \
    --out assets/norm_stats_handrecon_v2.json \
    --stats-dir assets/norm_stats_parts \
    --recompute
```

## 3. 关键超参（`configs/train_handrecon.yaml`）

| 项 | 值 | 备注 |
|---|---|---|
| total_steps | 25,000 | 预训练权重 finetune，25k 步足够（实际已训 ~30k） |
| batch_size | 200 / 卡（×4 卡 = 800） | |
| lr / weight_decay | 1e-4 / 0 | AdamW，betas (0.9, 0.999)；warmup 后 cosine 衰减到 `lr × lr_min_ratio`（默认 0） |
| warmup_steps | 2,500 | total_steps 的 10% |
| grad_clip | 1.0 | |
| lambda_v / lambda_3d / lambda_2d | 1.0 / 20.0 / 1.0 | 3D 损失与 2D 重投影损失权重（λ2d=0 关闭；像素 L1/image_size，(1−t) 加权） |
| ema_start_step / decay | 12,500 / 0.999 | total_steps 的 50%（论文同比例） |
| bf16 | true | |
| seed | 42 | |
| pretrained | `/root/code/vepfs/HUG-for-Recon-Gen/hug_checkpoint/hug_full.safetensors` | HUG 官方预训练权重（EMA，85K 步），仅加载模型参数做 finetune：251/483 张量载入，其余为冻结的 DINOv2 图像编码器（从 HF 加载），不含 optimizer/step |
| log / val / ckpt 间隔 | 20 / 1,000 / 1,000 | val 与 ckpt 对齐，续训点即评估点 |
| n_points_input | 4096 | query 点数 |
| pcl_crop_radius | 0.2 | 手比物体小，0.3→0.2 让点云密集覆盖手部 |

模型结构：RGB 用冻结 DINOv2-base（带 registers），点云用可训练 PointNeXt
（`pcl_width=64`，SA radii `[0.025, 0.05, 0.10, 0.20]`），
fusion transformer（`d_fusion=1024, 4 层, 8 头, 256 patches`），
flow transformer（`d_model=512, 6 层, 8 头, 50 步采样`）。
模态开关与论文 full model 一致：`use_rgb + use_depth + pointpainting`。

## 4. 验证集与指标（真实采样口径，`trainer.val` 段）

**验证/选型集 = DexYCB 官方 s0_val + HO3D 官方 evaluation split**（后者跨主体，
用于确保选出的模型泛化到未见过的主体）。每次验证：

- 走真实推理路径 `sample()`（50 步 ODE 完整采样），**不算 loss**（不反向传播、
  不参与选型，只无谓开销）
- 两种 GT schema 自动路由：DexYCB（MANO GT）-> `build_loss_dicts`；
  HO3D eval（joints/verts GT，无 MANO）-> `mano_forward` 比对
- 指标（`src/metrics.py`，相机系、单位 mm）：**MPJPE / PA-MPJPE**（21 关节）、
  **MPVPE / PA-MPVPE**（778 顶点）；PA 即 Procrustes 对齐
- 各数据集按占比等距采样共 `max_samples=4096` 条，确定性、跨 checkpoint 可比
- **多卡分片并行**：每 rank 用 `DistributedSampler` 评估自己的 shard 后
  all_reduce 聚合（rank-0 单卡跑采样式 val 会超过 NCCL 默认 600s 看门狗
  导致全任务崩溃；NCCL 超时已放宽到 30min 兜底 vepfs 慢写）
- EMA 启动后（`ema_start_step`）验证的是 **EMA 权重**（部署用的就是它），
  之前验证原始模型

**best 模型保存判据**：各数据集 `0.5×(PA-MPJPE + PA-MPVPE)` 后**等权平均**为
score，创新低即保存 `model_best.pt`（等权防止被样本量大的 DexYCB 主导）。

历史教训（已修复）：

- 旧 val 走 `forward()` 的"随机 t 单步恢复 x0"近似指标，约 2 倍乐观且尺度
  噪声大，导致 best 选择失真（曾挑中 9k 步的 checkpoint）。checkpoint 里的
  `val_metric` 版本标记保证 resume 时 best_val 自动重置、口径不混
- 旧逻辑在 EMA 启动前保存 best 时，"ema" 字段是未更新的预训练初始权重；
  现在未启动时存 `null`（评测端自动回退到 model 权重）

## 4b. 官方测试集全量评测（`src/eval_test.py`）

训练中的 val 用于 best checkpoint 选择；最终指标用官方测试集**全量**评测
（`torchrun --nproc_per_node=4 -m src.eval_test --config configs/train_handrecon.yaml`，
数据位置在 `trainer.test` 段配置）：

- **DexYCB s0_test**（76,529 条）：转换产物带完整 MANO GT，
  `sample()`（50 步 ODE 完整采样）-> `build_loss_dicts()` -> 四项指标
- **HO3D_v3 官方 evaluation split**（17,224 条）：只有 joints/verts GT
  （官方就不发布 MANO），`sample()` -> `mano_forward()` -> 与
  `joints_gt/verts_gt` 比对，同样四项指标。两条路径按 batch 内有无
  `mano_params` 字段自动切换
- 清单：`splits/dexycb_test.clean.txt` / `splits/ho3d_eval.clean.txt`
  （`make_handrecon_splits.py` + `filter_empty_masks.py --lists
  dexycb_test,ho3d_eval` 生成，剔除规则见第 2 节）
- 多卡分片：`torchrun --nproc_per_node=N`，指标经 all_reduce 聚合
- 默认评估 `<output_dir>/model_best.pt` 的 **EMA** 权重，结果表格打印并写
  `<output_dir>/test_results.json`；`--ckpt/--weights/--sets/--steps/
  --batch-size/--limit` 可覆盖（`--limit` 仅冒烟用，正式评测勿加）
- 冒烟：`python -m src.eval_test --config configs/train_handrecon.yaml
  --ckpt <任意ckpt> --steps 2 --limit 32`（随机权重已验证两条路径跑通）

## 5. 输出与恢复

### 日志文件（训练启动即创建）

`train.log_file` 指定结构化训练指标 JSONL；每条记录包含
`schema_version`、`event`、`run_id`、UTC 时间、rank/world_size 和 step。
此外，`output_dir/logs/` 下每个 rank 都有：

- `rank-<rank>.log`：该进程的完整 INFO/WARNING/ERROR 文本日志
- `rank-<rank>.error.log`：仅 ERROR 及以上，保留完整 traceback
- JSONL：rank 0 的 startup、配置、数据集、heartbeat、train、validation、checkpoint
  和 run_finished 事件，写入后立即 flush；`HUG_LOG_FSYNC=1` 可进一步启用 fsync

Python 未捕获异常会写入对应 rank 的 error 日志后重新抛出。OOM、SIGKILL、NCCL
原生 abort 等 Python 无法捕获的错误，还要保留 torchrun launcher 日志。
推荐在 tmux 中使用 hug 环境的 torchrun：

```bash
RUN=/root/code/vepfs/HUG-for-Recon-Gen/hand_recon/20260902_v6_canonical
mkdir -p "$RUN/logs/torchrun"
set -o pipefail
/root/code/vepfs/miniconda3/envs/hug/bin/torchrun \
  --log-dir "$RUN/logs/torchrun" --redirects 3 --tee 3 \
  --nproc_per_node=4 -m src.train --config configs/train_handrecon.yaml \
  2>&1 | tee "$RUN/logs/launcher.log"
```

`--tee 3` 同时保留终端和 worker 输出；`set -o pipefail` 确保 worker 失败时 shell
不会被 `tee` 错误地报告为成功。


## 6. 已知注意事项

- **磁盘**：根分区仅 20G，曾因旧输出写满导致 `torch.save` ENOSPC 崩溃。
  已清理（旧 `outputs/hug_train`/`smoke_test` 共 7.4G），且 `output_dir`
  与日志已迁到 vepfs 大文件系统；checkpoint 保存时剥离冻结 DINOv2
  （单文件 2.5G -> ~0.9G）。后续别把大文件写回根分区。
- 深度图单位 1mm uint16；HO3D 深度解码系数见 `scripts/README.md`。
- **DexYCB 左右手**：转换代码使用 `/root/code/vepfs/GPGFormer/weights/mano/MANO_LEFT.pkl`
  解码左手 PCA，并将左手 RGB/深度/相机坐标 canonicalize 为右手；右手模型下游
  不需要 side 分支。已有旧 pkl 缺少 side，不能原地修复，必须从 raw 全量重转换。
- 转换后的 parity audit（canonical DexYCB v2）：2000 样本 2D internal max=0；
  左手 canonical 3D mean 7.2mm / P90 12.3mm，右手 mean 2.8mm / P90 5.0mm。
  正式重训前应运行 `validate_dexycb_conversion.py` 和 `visualize_dataloader.py`。
- **当前混合训练配置**使用 `/root/code/vepfs/dataset/hand_recon_hug/dexycb_v2_canonical_right`
  与 `splits_v2/`，统计文件为 `assets/norm_stats_handrecon_v2.json`；不要混用旧
  `dexycb/` 和旧 checkpoint。
- **HO3D eval 关节顺序**：官方 `evaluation_xyz.json` 是 MANO 原始运动学
  顺序 `[腕, 食×3, 中×3, 小×3, 环×3, 拇×3, 指尖×5(拇,食,中,环,小)]`，
  和本仓库的 manotorch 顺序 `[腕, 拇×4, 食×4, 中×4, 环×4, 小×4]` 不同。
  `GraspDataset` 加载时按 `HO3D_RAW_TO_STD` 重排（曾因顺序错位导致
  PA-MPJPE 虚高至 40mm ≈ 均值姿势基线）。转换产物存官方原始顺序，勿在
  转换脚本里排。
- HO3D 评估集（`--split evaluation` 产物）无 MANO 标注、无分割 mask，
  不在训练 loop 内，只用于第 4b 节的推理评测（`GraspDataset` 对空
  `object_mask` 已做兼容：有 `condition_point` 就不解码 mask）。
