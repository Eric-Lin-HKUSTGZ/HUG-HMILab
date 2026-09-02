"""Evaluate the HUG hand-reconstruction checkpoint on DexYCB and HO3D.

This evaluator is intentionally separate from the grasp-generation evaluator:
it consumes the reconstruction pkl schema and handles the HO3D evaluation
schema (joints/vertices GT without MANO parameters).
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader, DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataloader.grasp_dataset import GraspDataset
from src.inference import load_model
from src.metrics import joint_mesh_errors
from src.train import is_main, setup_ddp

console = Console()
METRIC_KEYS = ("mpjpe", "pa_mpjpe", "mpvpe", "pa_mpvpe")

# Official HO3D evaluation files use MANO's raw kinematic order.  The model
# and all converted training data use the standard [wrist, thumb, index,
# middle, ring, pinky] order.
HO3D_RAW_TO_STD = [
    0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20
]


class ReconDataset(GraspDataset):
    """GraspDataset plus joints/vertices-only GT for HO3D evaluation pkls."""

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        path = self.grasp_files[idx]
        with open(path, "rb") as f:
            data = pickle.load(f)
        if "joints_gt" in data:
            joints = torch.from_numpy(
                np.asarray(data["joints_gt"], dtype="float32")[HO3D_RAW_TO_STD]
            )
            verts = torch.from_numpy(
                np.asarray(data["verts_gt"], dtype="float32")
            )
            out["joints_gt"] = joints
            out["verts_gt"] = verts
        return out


def evaluate_dataset(name, model, loader, device, rank, world_size, bf16, steps):
    model.eval()
    sums = {k: 0.0 for k in METRIC_KEYS}
    n = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=(bf16 and device.type == "cuda")
            ):
                samples = model.sample(
                    point_uv=batch["point_uv"].to(device, non_blocking=True),
                    camera_K=batch["camera_K"].to(device, non_blocking=True),
                    steps=steps,
                    rgb=batch["rgb"].to(device, non_blocking=True)
                    if "rgb" in batch
                    else None,
                    pcl_xyz=batch["pcl_xyz"].to(device, non_blocking=True)
                    if "pcl_xyz" in batch
                    else None,
                    pcl_rgb=batch["pcl_rgb"].to(device, non_blocking=True)
                    if "pcl_rgb" in batch
                    else None,
                )
                if "mano_params" in batch:
                    preds, targets = model.build_loss_dicts(
                        samples, batch["mano_params"].to(device)
                    )
                    errs = joint_mesh_errors(
                        preds["landmarks_3d"].float(),
                        targets["landmarks_3d"].float(),
                        preds["vertices"].float(),
                        targets["vertices"].float(),
                    )
                else:
                    betas = model.fixed_betas.expand(samples.shape[0], -1)
                    pred = model.mano_forward(samples, betas=betas)
                    errs = joint_mesh_errors(
                        pred["landmarks_3d"].float(),
                        batch["joints_gt"].to(device).float(),
                        pred["vertices"].float(),
                        batch["verts_gt"].to(device).float(),
                    )
            for key in METRIC_KEYS:
                sums[key] += errs[key].float().sum().item()
            n += int(errs["mpjpe"].shape[0])
            if is_main(rank) and (batch_idx + 1) % 50 == 0:
                console.print(
                    f"  [{name}] batch {batch_idx + 1}/{len(loader)}, "
                    f"{n} local samples, {time.perf_counter() - t0:.0f}s"
                )

    values = torch.tensor(
        [sums[k] for k in METRIC_KEYS] + [float(n)], device=device, dtype=torch.float64
    )
    if world_size > 1:
        torch.distributed.all_reduce(values)
    total_n = max(values[-1].item(), 1.0)
    result = {key: float(values[i].item() / total_n) for i, key in enumerate(METRIC_KEYS)}
    result["n_samples"] = int(values[-1].item())
    result["seconds"] = round(time.perf_counter() - t0, 1)
    return result


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
    output: Path = Path(
        "/root/code/vepfs/HUG-for-Recon-Gen/hand_recon/20260902_v6_canonical/test_results.json"
    ),
    weights: str = "ema",
    steps: int = 50,
    batch_size: int = 64,
    num_workers: int = 4,
    limit: Optional[int] = None,
):
    rank, _local_rank, world_size, device = setup_ddp()
    cfg_ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(cfg_ckpt["cfg"])
    # load_model restores the same embedded cfg/norm_stats and loads EMA/model
    model = load_model(checkpoint, use_ema=(weights == "ema"), device=str(device))
    model.eval()
    common = dict(
        split="val",
        n_points_input=int(cfg.trainer.data.get("n_points_input", 4096)),
        pcl_crop_radius=float(cfg.trainer.model.get("pcl_crop_radius", 0.2)),
        use_rgb=bool(cfg.trainer.model.get("use_rgb", True)),
        use_depth=bool(cfg.trainer.model.get("use_depth", True)),
    )
    bf16 = bool(cfg.trainer.train.get("bf16", True)) and device.type == "cuda"
    datasets = [
        ("dexycb_test", dexycb_path, dexycb_samples),
        ("ho3d_eval", ho3d_path, ho3d_samples),
    ]
    results = {}
    for name, path, samples_file in datasets:
        ds = ReconDataset(str(path), samples_filename=str(samples_file), **common)
        if limit is not None:
            ds.grasp_files = ds.grasp_files[: int(limit)]
        sampler = (
            DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False)
            if world_size > 1
            else None
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        if is_main(rank):
            console.print(f"[cyan]== {name}: {len(ds)} samples ==[/cyan]")
        results[name] = evaluate_dataset(
            name, model, loader, device, rank, world_size, bf16, steps
        )
        if world_size > 1:
            torch.distributed.barrier()

    if is_main(rank):
        table = Table(title="HUG hand reconstruction test (mm)")
        table.add_column("dataset")
        table.add_column("n", justify="right")
        for key in METRIC_KEYS:
            table.add_column(key.upper(), justify="right")
        for name, result in results.items():
            table.add_row(
                name,
                str(result["n_samples"]),
                *[f"{result[key]:.2f}" for key in METRIC_KEYS],
            )
        console.print(table)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(checkpoint),
            "weights": weights,
            "steps": steps,
            "results": results,
        }
        output.write_text(json.dumps(payload, indent=2) + "\n")
        console.print(f"[green]wrote {output}[/green]")
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    tyro.cli(main)
