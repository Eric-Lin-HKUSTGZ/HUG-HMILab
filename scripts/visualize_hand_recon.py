"""GPGFormer-style visualisation for HUG hand reconstruction.

Each output contains the RGB image with projected GT/predicted meshes and a
root-relative 3D GT/prediction mesh comparison.  This is deliberately not the
HUG grasp-generation six-panel renderer.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import cv2
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import matplotlib
except ModuleNotFoundError:
    # The lightweight HUG environment does not install matplotlib.  The
    # server's pose environment already provides it; append that site-packages
    # directory so HUG's NumPy/Torch versions keep precedence.
    import sys

    sys.path.append("/root/code/vepfs/miniconda3/envs/pose/lib/python3.10/site-packages")
    import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.dataloader.grasp_dataset import GraspDataset
from src.inference import load_model
from src.metrics import joint_mesh_errors

HO3D_RAW_TO_STD = [
    0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20
]
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def decode_rgb(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def project(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = np.maximum(points[:, 2:3], 1e-6)
    q = points @ K.T
    return q[:, :2] / z


def add_mesh_2d(ax, verts, faces, K, color, alpha):
    uv = project(verts, K)
    tris = uv[faces]
    z = verts[:, 2]
    valid = np.all(np.isfinite(tris), axis=(1, 2)) & (z[faces].mean(axis=1) > 0)
    if np.any(valid):
        ax.add_collection(
            PolyCollection(
                tris[valid], facecolors=color, edgecolors="none", alpha=alpha
            )
        )


def draw_edges(ax, joints, color, linewidth=1.4, alpha=0.9):
    uv = joints
    for a, b in HAND_EDGES:
        ax.plot(
            [uv[a, 0], uv[b, 0]],
            [uv[a, 1], uv[b, 1]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )


def set_equal_3d(ax, xyz):
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    center = (lo + hi) / 2.0
    radius = max(float((hi - lo).max()) / 2.0, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def render_one(out_path, stem, rgb, K, pred_v, pred_j, gt_v, gt_j, faces, metrics):
    gt_uv = project(gt_j, K)
    pred_uv = project(pred_j, K)
    fig = plt.figure(figsize=(16, 7.2), dpi=140)
    gs = fig.add_gridspec(1, 2, width_ratios=(0.95, 1.45), wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgb)
    add_mesh_2d(ax, gt_v, faces, K, "#16a34a", 0.20)
    add_mesh_2d(ax, pred_v, faces, K, "#ef4444", 0.20)
    draw_edges(ax, gt_uv, "#16a34a", 1.3)
    draw_edges(ax, pred_uv, "#ef4444", 1.3)
    ax.scatter(gt_uv[:, 0], gt_uv[:, 1], s=12, c="#16a34a", label="GT", zorder=4)
    ax.scatter(pred_uv[:, 0], pred_uv[:, 1], s=12, c="#ef4444", label="Pred", zorder=5)
    ax.set_title(f"2D projection | MPJPE={metrics['mpjpe']:.2f}mm")
    ax.set_xlim(0, rgb.shape[1])
    ax.set_ylim(rgb.shape[0], 0)
    ax.set_axis_off()
    ax.legend(loc="lower right", framealpha=0.9)

    ax3 = fig.add_subplot(gs[0, 1], projection="3d")
    root_gt, root_pred = gt_j[0], pred_j[0]
    gt_vr, pred_vr = gt_v - root_gt, pred_v - root_pred
    gt_jr, pred_jr = gt_j - root_gt, pred_j - root_pred
    ax3.add_collection3d(
        Poly3DCollection(
            gt_vr[faces], facecolors="#16a34a", edgecolors="none", alpha=0.20,
            label="GT mesh"
        )
    )
    ax3.add_collection3d(
        Poly3DCollection(
            pred_vr[faces], facecolors="#ef4444", edgecolors="none", alpha=0.20,
            label="Pred mesh"
        )
    )
    for joints, color in ((gt_jr, "#16a34a"), (pred_jr, "#ef4444")):
        for a, b in HAND_EDGES:
            ax3.plot(
                [joints[a, 0], joints[b, 0]],
                [joints[a, 1], joints[b, 1]],
                [joints[a, 2], joints[b, 2]], color=color, linewidth=1.5,
            )
        ax3.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=12, c=color)
    all_xyz = np.concatenate([gt_vr, pred_vr], axis=0)
    set_equal_3d(ax3, all_xyz)
    ax3.view_init(elev=18, azim=-68)
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_zlabel("Z (m)")
    ax3.set_title(
        "3D root-relative mesh | "
        f"MPVPE={metrics['mpvpe']:.2f}mm | PA-MPJPE={metrics['pa_mpjpe']:.2f}mm | "
        f"PA-MPVPE={metrics['pa_mpvpe']:.2f}mm"
    )
    from matplotlib.lines import Line2D

    ax3.legend(
        handles=[
            Line2D([0], [0], color="#16a34a", lw=4, label="GT"),
            Line2D([0], [0], color="#ef4444", lw=4, label="Pred"),
        ],
        loc="upper right",
    )
    fig.suptitle(stem, fontsize=13, y=0.99)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(
    checkpoint: Path = Path(
        "/root/code/vepfs/HUG-for-Recon-Gen/hand_recon/20260902_v6_canonical/model_best.pt"
    ),
    dexycb_path: Path = Path(
        "/root/code/vepfs/dataset/hand_recon_hug/dexycb_v2_canonical_right"
    ),
    dexycb_samples: Path = Path(
        "/root/code/vepfs/dataset/hand_recon_hug/splits_v2/dexycb_test.clean.txt"
    ),
    ho3d_path: Path = Path("/root/code/vepfs/dataset/hand_recon_hug/ho3d_eval"),
    ho3d_samples: Path = Path(
        "/root/code/vepfs/dataset/hand_recon_hug/splits_v2/ho3d_eval.clean.txt"
    ),
    output_dir: Path = Path(
        "/root/code/vepfs/HUG-for-Recon-Gen/hand_recon/20260902_v6_canonical/vis_mesh"
    ),
    num_samples: int = 16,
    batch_size: int = 4,
    weights: str = "ema",
    steps: int = 50,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(checkpoint, use_ema=(weights == "ema"), device=device)
    model.eval()
    cfg = OmegaConf.create(torch.load(checkpoint, map_location="cpu", weights_only=False)["cfg"])
    common = dict(
        split="val",
        n_points_input=int(cfg.trainer.data.get("n_points_input", 4096)),
        pcl_crop_radius=float(cfg.trainer.model.get("pcl_crop_radius", 0.2)),
        use_rgb=bool(cfg.trainer.model.get("use_rgb", True)),
        use_depth=bool(cfg.trainer.model.get("use_depth", True)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"checkpoint": str(checkpoint), "weights": weights, "steps": steps, "datasets": {}}
    for name, data_path, samples_file in (
        ("dexycb", dexycb_path, dexycb_samples),
        ("ho3d", ho3d_path, ho3d_samples),
    ):
        ds = GraspDataset(str(data_path), samples_filename=str(samples_file), **common)
        if len(ds) > num_samples:
            idx = np.linspace(0, len(ds) - 1, num_samples, dtype=int).tolist()
            ds.grasp_files = [ds.grasp_files[i] for i in idx]
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
        ds_out = output_dir / name
        ds_out.mkdir(parents=True, exist_ok=True)
        entries = []
        with torch.no_grad():
            for batch in loader:
                inputs = dict(
                    point_uv=batch["point_uv"].to(device),
                    camera_K=batch["camera_K"].to(device),
                    steps=steps,
                    rgb=batch["rgb"].to(device) if "rgb" in batch else None,
                    pcl_xyz=batch["pcl_xyz"].to(device) if "pcl_xyz" in batch else None,
                    pcl_rgb=batch["pcl_rgb"].to(device) if "pcl_rgb" in batch else None,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    samples = model.sample(**inputs)
                pred = model.mano_forward(samples, betas=model.fixed_betas.expand(samples.shape[0], -1))
                for i, stem in enumerate(batch["stem"]):
                    raw_path = data_path / f"{stem}.pkl"
                    with open(raw_path, "rb") as f:
                        raw = pickle.load(f)
                    if "grasp" in raw and raw["grasp"] is not None:
                        gt_j = np.asarray(raw["grasp"]["landmarks_3d"], np.float32)
                        gt_v = np.asarray(raw["grasp"]["mesh_vertices"], np.float32)
                    else:
                        gt_j = np.asarray(raw["joints_gt"], np.float32)[HO3D_RAW_TO_STD]
                        gt_v = np.asarray(raw["verts_gt"], np.float32)
                    pred_j = pred["landmarks_3d"][i].float().cpu().numpy()
                    pred_v = pred["vertices"][i].float().cpu().numpy()
                    gt_j_t, gt_v_t = torch.from_numpy(gt_j)[None], torch.from_numpy(gt_v)[None]
                    err = joint_mesh_errors(
                        torch.from_numpy(pred_j)[None], gt_j_t,
                        torch.from_numpy(pred_v)[None], gt_v_t,
                    )
                    metrics = {k: float(v[0]) for k, v in err.items()}
                    image = decode_rgb(raw["image"])
                    K = np.asarray(raw["camera"]["K"], np.float32)
                    safe_stem = str(stem).replace("/", "__")
                    render_one(ds_out / f"{safe_stem}.png", safe_stem, image, K, pred_v, pred_j, gt_v, gt_j, model.mesh_faces, metrics)
                    entries.append({"stem": str(stem), **metrics})
        summary["datasets"][name] = {"n_visualized": len(entries), "samples": entries}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
