# scripts/ — 数据准备与可视化脚本

本目录包含 1M-HUGs 抓取生成训练的数据准备工具与预测可视化脚本，产物由
`src/dataloader/grasp_dataset.py` 直接消费。

## 文件一览

| 文件 | 作用 |
|---|---|
| `convert_dinov2.py` | 官方 DINOv2 `.pth` → HuggingFace 格式（本地加载，免联网） |
| `make_val_split.py` | 为 1M-HUGS 生成录制级（recording-level）验证集划分 |
| `render_predictions.py` | 将 `grasp_pred/` 预测渲染为静态 PNG 六联图 |

## 各脚本说明

### `convert_dinov2.py`

将官方 DINOv2 ViT-B/14（带 registers）`.pth` 权重转换为 HuggingFace
`Dinov2Model` 布局（`config.json` + `pytorch_model.bin`），供模型
`encoder_name` 指向本地目录加载。输入输出路径硬编码在脚本头部，使用前请
按需修改：

```bash
python scripts/convert_dinov2.py
# → <OUT>/pytorch_model.bin
```

### `make_val_split.py`

为 1M-HUGS 训练数据生成**录制级**验证集划分。之所以按录制划分：每次物理
抓取会产生数百个 (frame, grasp) 样本（含同时间戳的 `_grayscale` 孪生帧），
随机按帧划分会造成数据泄漏。按录制（stem 去掉 `<frame>_<hash>[_grayscale]`
后缀）分组，保证同一次抓取的所有帧落在同一侧。

```bash
python scripts/make_val_split.py \
    --dataset-path /root/code/vepfs/dataset/1m-hugs/grasp_data \
    --n-recordings 48 [--seed 42]
```

输出 `{dataset_path}/split_val.txt`（每行一个 stem），训练配置的
`val_split_file` 引用该文件。需先构建 `samples.txt` 索引（`GraspDataset`
首次扫描数据目录时会自动生成）。

### `render_predictions.py`

对 `grasp_pred/` 中的预测结果离线渲染：每个样本生成一张六联图（RGB+2D
joints、depth、front/side/angled 3D hand views 和信息面板）。

```bash
python scripts/render_predictions.py \
    --dataset-path data/hug_bench \
    --output-dir prediction_images
```

## 依赖

`numpy`、`opencv-python`（cv2）、`tyro`（CLI）、`rich`（日志）、`torch`。
