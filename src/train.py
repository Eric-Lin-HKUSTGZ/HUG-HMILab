"""DDP training for HUG (GraspFlowModel), configured by a YAML file.

Launch (multi-GPU):
    torchrun --nproc_per_node=4 -m hug.train --config configs/train_hug.yaml

Smoke test (short run on a subset):
    torchrun --nproc_per_node=4 -m hug.train --config configs/train_hug.yaml \
        --max-steps 30 --max-train-samples 20000

Loss (paper §4.2, Eq. 1 + optional 2D reprojection term):
    L = λv * Lv + λ3D * (1 - t) * L3D + λ2D * (1 - t) * L2D
  Lv  : velocity MSE in normalized 99D space (from model.forward preds/targets)
  L3D : L1 on MANO landmarks in the camera frame, weighted per-sample by (1-t)
  L2D : L1 on MANO joints reprojected to 2D with camera_K vs GT landmarks_2d,
        normalized by image size, weighted per-sample by (1-t); λ2d=0 disables

LR schedule: linear warmup (warmup_steps) -> cosine decay to lr * lr_min_ratio.

Checkpoints embed cfg + norm_stats + EMA weights in the exact layout that
`src/inference.py:load_model` consumes ({model, ema, cfg, norm_stats, ...}),
so a training output dir can be pointed at inference/app directly.
"""

import faulthandler
import json
import logging
import math
import os
import socket
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import tyro
from omegaconf import OmegaConf
from rich.console import Console
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, DistributedSampler

from .dataloader.grasp_dataset import GraspDataset
from .metrics import joint_mesh_errors
from .models.grasp_model import GraspFlowModel
from .utils.data_keys import NORM_STATS_FILE

logger = logging.getLogger("hug.train")
console = Console()

LOG_SCHEMA_VERSION = "train-log-v2"


class ContextFormatter(logging.Formatter):
    """Add DDP/process context to every durable text-log record."""

    def format(self, record):
        base = super().format(record)
        return (
            f"[rank={getattr(record, 'rank', os.environ.get('RANK', '0'))} "
            f"local_rank={getattr(record, 'local_rank', os.environ.get('LOCAL_RANK', '0'))} "
            f"pid={os.getpid()}] {base}"
        )


class JsonlWriter:
    """Flush-on-write structured event writer; only rank 0 owns metrics JSONL."""

    def __init__(self, path: Path, run_id: str, rank: int, world_size: int):
        self.path = path
        self.run_id = run_id
        self.rank = rank
        self.world_size = world_size
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: str, payload: dict | None = None, step: int | None = None):
        record = {
            "schema_version": LOG_SCHEMA_VERSION,
            "event": event,
            "run_id": self.run_id,
            "time": datetime.now(timezone.utc).isoformat(),
            "rank": self.rank,
            "world_size": self.world_size,
        }
        if step is not None:
            record["step"] = int(step)
        if payload:
            record.update(payload)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            if os.environ.get("HUG_LOG_FSYNC", "0") == "1":
                os.fsync(f.fileno())


def _resolve_log_path(train_cfg, out_dir: Path) -> Path:
    log_file = train_cfg.get("log_file")
    return (
        Path(log_file)
        if log_file and Path(log_file).is_absolute()
        else out_dir / (log_file or "train_log.jsonl")
    )


def configure_logging(out_dir: Path, log_path: Path, rank: int, local_rank: int, world_size: int):
    """Install durable rank-aware logging before data/model initialization."""
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    text_path = log_dir / f"rank-{rank}.log"
    error_path = log_dir / f"rank-{rank}.error.log"
    fmt = ContextFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    for path, level in ((text_path, logging.INFO), (error_path, logging.ERROR)):
        handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(fmt)
        root.addHandler(handler)
    stream = logging.StreamHandler(sys.stderr)
    stream.setLevel(logging.INFO)
    stream.setFormatter(fmt)
    root.addHandler(stream)
    logging.captureWarnings(True)
    faulthandler.enable()
    run_id = os.environ.get("TORCHELASTIC_RUN_ID") or uuid.uuid4().hex[:12]
    writer = JsonlWriter(log_path, run_id, rank, world_size) if is_main(rank) else None
    context = {"rank": rank, "local_rank": local_rank, "world_size": world_size}
    logger.info(
        "logging initialized: output=%s metrics=%s text=%s errors=%s host=%s",
        out_dir, log_path, text_path, error_path, socket.gethostname(), extra=context,
    )
    return writer, context


def _safe_event(writer, logger_obj, event, payload=None, step=None):
    """Write an event without hiding the original failure if logging breaks."""
    if writer is None:
        return
    try:
        writer.write(event, payload, step=step)
    except Exception:
        logger_obj.exception("failed to write structured event %s", event)


def _phase_log(context, phase, message, *args):
    logger.info("phase=%s %s", phase, message % args if args else message, extra=context)


def install_exception_hooks(context, state=None):
    """Persist uncaught main/thread exceptions with full tracebacks."""
    state = state if state is not None else {}

    def handle(exc_type, exc_value, exc_tb):
        context_now = {
            **context,
            "step": state.get("step"),
            "phase": state.get("phase"),
        }
        if issubclass(exc_type, KeyboardInterrupt):
            logger.error("interrupted", exc_info=(exc_type, exc_value, exc_tb), extra=context_now)
        else:
            logger.critical("uncaught exception", exc_info=(exc_type, exc_value, exc_tb), extra=context_now)
        logging.shutdown()
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = handle
    if hasattr(threading, "excepthook"):
        def thread_hook(args):
            context_now = {
                **context,
                "step": state.get("step"),
                "phase": state.get("phase"),
            }
            logger.critical("uncaught thread exception", exc_info=(args.exc_type, args.exc_value, args.exc_traceback), extra=context_now)
            logging.shutdown()
        threading.excepthook = thread_hook



# --------------------------------------------------------------------------
# DDP helpers
# --------------------------------------------------------------------------

def setup_ddp() -> tuple[int, int, int, torch.device]:
    """Init process group from torchrun env vars. Returns (rank, local_rank, world_size, device)."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and not dist.is_initialized():
        # 30min watchdog: val sampling + vepfs checkpoint writes can exceed the
        # default 10min NCCL timeout
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return rank, local_rank, world_size, device


def is_main(rank: int) -> bool:
    return rank == 0


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def build_datasets(cfg):
    """Train dataset + val dataset.

    Single dataset at `trainer.data.dataset_path` + recording-level
    `val_split_file` (built by scripts/make_val_split.py), falling back to
    a deterministic random frame-level split.
    """
    data_cfg = cfg.trainer.data
    common = dict(
        split="train",
        n_points_input=data_cfg.n_points_input,
        pcl_crop_radius=cfg.trainer.model.get("pcl_crop_radius", 0.3),
        use_rgb=cfg.trainer.model.get("use_rgb", True),
        use_depth=cfg.trainer.model.get("use_depth", True),
    )

    dataset_path = str(data_cfg.dataset_path)
    full = GraspDataset(
        dataset_path, samples_filename=data_cfg.get("samples_filename"), **common
    )
    n = len(full)
    root = Path(dataset_path)

    split_file = root / data_cfg.get("val_split_file", "split_val.txt")
    if split_file.exists():
        val_stems = set(split_file.read_text().splitlines())
        val_idx, train_idx = [], []
        for i, p in enumerate(full.grasp_files):
            stem = p.relative_to(root).with_suffix("").as_posix()
            (val_idx if stem in val_stems else train_idx).append(i)
        logger.info(
            f"Recording-level split from {split_file.name}: "
            f"{len(val_idx)} val / {len(train_idx)} train frames"
        )
    else:
        logger.warning(
            f"{split_file} not found — falling back to random frame-level split "
            f"(leaks across frames of one grasp; run scripts/make_val_split.py)"
        )
        rng = np.random.default_rng(42)
        val_count = min(int(data_cfg.get("val_count", 512)), n // 10)
        val_idx = sorted(rng.choice(n, size=val_count, replace=False).tolist())
        val_set = set(val_idx)
        train_idx = [i for i in range(n) if i not in val_set]

    train_ds = GraspDataset(dataset_path, indices=train_idx, **common)
    val_ds = GraspDataset(dataset_path, indices=val_idx, **common)
    return train_ds, val_ds


def build_loaders(cfg, train_ds, val_ds, rank, world_size, max_train_samples=None):
    train_cfg = cfg.trainer.train
    if max_train_samples is not None and hasattr(train_ds, "grasp_files"):
        train_ds.grasp_files = train_ds.grasp_files[:max_train_samples]
    sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.trainer.data.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=cfg.trainer.data.prefetch_factor,
        drop_last=True,
    )
    # Val is sharded across ranks (see run_val); a plain loader suffices.
    max_val = cfg.trainer.data.get("max_val_samples")
    if max_val is not None and len(val_ds) > int(max_val):
        from torch.utils.data import Subset

        # 等距采样整个 val 集，采样确定、跨 checkpoint 可比
        n = len(val_ds)
        k = int(max_val)
        val_ds = Subset(val_ds, sorted({round(i * n / k) for i in range(k)}))
        logger.info(f"val subsample -> {len(val_ds)} samples")
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        sampler=(
            DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
            if world_size > 1
            else None
        ),
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, [("val", val_loader)], sampler


def infinite_loader(loader, sampler):
    """Yield batches forever, re-shuffling at every pass (step-based training)."""
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


# --------------------------------------------------------------------------
# Loss (paper Eq. 1)
# --------------------------------------------------------------------------

def compute_loss(preds, targets, time_weight, lambda_v, lambda_3d, lambda_2d=0.0):
    """L = λv·Lv + λ3D·(1−t)·L3D + λ2D·(1−t)·L2D. Returns (loss, {component: float}).

    Losses are computed in fp32: under bf16 autocast the DiT velocity output is
    bf16 while the flow target (eps - x_norm) is fp32, and mixed-dtype
    MseLossBackward raises "Found dtype Float but expected BFloat16".

    L2D (optional, when lambda_2d > 0 and targets carry landmarks_2d):
    reprojection L1 between predicted MANO joints projected with camera_K and
    the GT 2D joints (same 224-crop pixel space), normalized by image size and
    weighted per-sample by (1-t) like L3D.
    """
    lv = F.mse_loss(
        preds["params_norm"].float(), targets["params_norm"].float()
    )
    l3d_per_sample = (
        (preds["landmarks_3d"].float() - targets["landmarks_3d"].float())
        .abs()
        .mean(dim=(1, 2))
    )  # (B,)
    l3d = (time_weight * l3d_per_sample).mean()
    loss = lambda_v * lv + lambda_3d * l3d
    comps = {
        "loss": loss.item(),
        "lv": lv.item(),
        "l3d": l3d.item(),
        "l3d_raw": l3d_per_sample.mean().item(),
    }

    if lambda_2d > 0.0 and "landmarks_2d" in targets:
        K = targets["camera_K"].float()  # (B, 3, 3)
        pred_j = preds["landmarks_3d"].float()  # (B, 21, 3), camera frame (with t)
        z = pred_j[..., 2].clamp(min=1e-3)
        pred_uv = torch.stack(
            [
                K[:, 0, 0].unsqueeze(1) * pred_j[..., 0] / z + K[:, 0, 2].unsqueeze(1),
                K[:, 1, 1].unsqueeze(1) * pred_j[..., 1] / z + K[:, 1, 2].unsqueeze(1),
            ],
            dim=-1,
        )  # (B, 21, 2)
        image_size = float(targets.get("image_size", 224))
        l2d_per_sample = (
            (pred_uv - targets["landmarks_2d"].float()).abs().mean(dim=(1, 2))
            / image_size
        )  # (B,)
        l2d = (time_weight * l2d_per_sample).mean()
        loss = loss + lambda_2d * l2d
        comps["loss"] = loss.item()
        comps["l2d"] = l2d.item()
        comps["l2d_px"] = l2d_per_sample.mean().item() * image_size

    return loss, comps


# --------------------------------------------------------------------------
# Validation (sharded across ranks on the unwrapped module; metrics
# all_reduced at the end of each dataset)
# --------------------------------------------------------------------------

METRIC_KEYS = ("mpjpe", "pa_mpjpe", "mpvpe", "pa_mpvpe")

# Version tag stored in checkpoints; a resume from a checkpoint saved with a
# different val metric (e.g. the old x0-recovery proxy, whose scale was ~2x
# optimistic) resets best_val so model_best.pt tracking stays meaningful.
VAL_METRIC = "sampling-mixed-v1"


def run_val(raw_model, val_loaders, device, bf16, rank, world_size):
    """Sampling-based validation: real inference quality, no loss.

    val_loaders: list of (name, DataLoader). With world_size > 1 each rank
    evaluates its DistributedSampler shard and metric sums are all_reduced.
    Each batch carries MANO GT -> build_loss_dicts; metrics are
    MPJPE / PA-MPJPE / MPVPE / PA-MPVPE in mm. Returns {name: {metric: mean_mm}}.
    """
    raw_model.eval()
    results = {}
    for name, loader in val_loaders:
        sums = {k: 0.0 for k in METRIC_KEYS}
        n = 0
        with torch.no_grad():
            for batch in loader:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                    samples = raw_model.sample(
                        point_uv=batch["point_uv"].to(device),
                        camera_K=batch["camera_K"].to(device),
                        rgb=batch["rgb"].to(device) if "rgb" in batch else None,
                        pcl_xyz=batch["pcl_xyz"].to(device) if "pcl_xyz" in batch else None,
                        pcl_rgb=batch["pcl_rgb"].to(device) if "pcl_rgb" in batch else None,
                    )
                    preds, targets = raw_model.build_loss_dicts(
                        samples, batch["mano_params"].to(device)
                    )
                    errs = joint_mesh_errors(
                        preds["landmarks_3d"].float(),
                        targets["landmarks_3d"].float(),
                        preds["vertices"].float(),
                        targets["vertices"].float(),
                    )
                for k in METRIC_KEYS:
                    sums[k] += errs[k].float().sum().item()
                n += errs["mpjpe"].shape[0]
        if world_size > 1:
            stats = torch.tensor(
                [sums[k] for k in METRIC_KEYS] + [float(n)], device=device
            )
            dist.all_reduce(stats)
            sums = {k: float(stats[i]) for i, k in enumerate(METRIC_KEYS)}
            n = float(stats[-1])
        results[name] = {k: v / max(n, 1) for k, v in sums.items()}
    raw_model.train()
    return results


# --------------------------------------------------------------------------
# Checkpoint
# --------------------------------------------------------------------------

def _strip_frozen_encoder(state):
    """Drop frozen DINOv2 tensors from a state dict (reloadable from HF at
    inference; inference.py:load_model builds the encoder then loads with
    strict=False). Cuts ~700MB from each saved copy."""
    return {k: v for k, v in state.items() if not k.startswith("image_encoder.")}


def save_checkpoint(path, raw_model, ema_model, optimizer, cfg, norm_stats, step, best_val=None):
    """Save in the exact layout src/inference.py:load_model consumes.

    EMA weights are only saved once averaging has actually started
    (ema_start_step); before that the AveragedModel still holds the init copy
    and saving it as "ema" would silently publish untrained weights (the
    eval path defaults to EMA). ema=None makes loaders fall back to "model".
    """
    ema_started = ema_model is not None and int(ema_model.n_averaged.item()) > 0
    ckpt = {
        "model": _strip_frozen_encoder(raw_model.state_dict()),
        "ema": _strip_frozen_encoder(ema_model.state_dict()) if ema_started else None,
        "optimizer": optimizer.state_dict(),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "norm_stats": norm_stats,
        "weights_kind": "model",
        "step": step,
        "best_val": best_val,
        "val_metric": VAL_METRIC,  # metric semantics for best_val comparisons
    }
    tmp = path.with_suffix(".tmp.pt")
    torch.save(ckpt, tmp)
    tmp.rename(path)


def load_pretrained(model, path, device):
    """Initialize `model` from HUG pretrained weights (safetensors or .pt).

    Unlike `train.resume` this only restores model weights - step/optimizer/
    best_val all start fresh (finetune setup). The released hug_full.safetensors
    stores EMA weights without the frozen DINOv2 image_encoder (loaded from HF
    at model build), so loading is strict=False and the loaded/missing counts
    are reported.
    """
    p = Path(path)
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file

        sd = load_file(str(p))
    else:
        sd = torch.load(str(p), map_location=device, weights_only=False)
        sd = sd.get("model", sd.get("ema", sd))
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}

    msd = model.state_dict()
    shape_mismatch = [k for k in sd if k in msd and sd[k].shape != msd[k].shape]
    if shape_mismatch:
        raise RuntimeError(f"pretrained shape mismatch: {shape_mismatch[:5]} ...")
    incompatible = model.load_state_dict(sd, strict=False)
    n_loaded = len([k for k in sd if k in msd])
    logging.info(
        f"pretrained <- {p}: loaded {n_loaded}/{len(msd)} tensors "
        f"(missing = frozen image_encoder + {len(incompatible.unexpected_keys)} unexpected)"
    )
    return n_loaded


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(
    config: Path,
    max_steps: Optional[int] = None,
    max_train_samples: Optional[int] = None,
) -> None:
    """Train HUG with DDP.

    Args:
        config: Path to the YAML config (see configs/train_hug.yaml).
        max_steps: Override trainer.train.total_steps (smoke tests).
        max_train_samples: Cap the train set size (smoke tests).
    """
    # Load config and initialize durable logs before DDP/data/model setup so
    # startup failures are persisted as well.
    cfg = OmegaConf.load(config)
    train_cfg = cfg.trainer.train
    if max_steps is not None:
        train_cfg.total_steps = max_steps
    out_dir = Path(train_cfg.output_dir)
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    log_path = _resolve_log_path(train_cfg, out_dir)
    writer, log_context = configure_logging(out_dir, log_path, rank, local_rank, world_size)
    exception_state = {"step": None, "phase": "startup"}
    install_exception_hooks(log_context, exception_state)
    if writer:
        writer.write("startup", {"config": str(config), "output_dir": str(out_dir)})

    rank, local_rank, world_size, device = setup_ddp()
    torch.manual_seed(train_cfg.seed + rank)

    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, out_dir / "config.yaml")
        console.print(f"[cyan]world_size={world_size}, output -> {out_dir}[/cyan]")
        if writer:
            writer.write("config_saved", {"config_copy": str(out_dir / "config.yaml")})

    logger.info("config loaded: %s", config, extra=log_context)
    if writer:
        writer.write("config_loaded", {"config": str(config)})
    norm_stats_file = Path(cfg.trainer.data.get("norm_stats_file", str(NORM_STATS_FILE)))
    with open(norm_stats_file) as f:
        norm_stats = json.load(f)
    logger.info("norm_stats loaded: %s", norm_stats_file, extra=log_context)
    if writer:
        writer.write("norm_stats_loaded", {"path": str(norm_stats_file)})
    if is_main(rank):
        console.print(f"[cyan]norm_stats: {norm_stats_file}[/cyan]")

    # ---- data ----
    train_ds, val_ds = build_datasets(cfg)
    train_loader, val_loaders, sampler = build_loaders(
        cfg, train_ds, val_ds, rank, world_size, max_train_samples
    )
    logger.info("datasets ready: train=%d val=%d", len(train_ds), len(val_ds), extra=log_context)
    if writer:
        writer.write("datasets_ready", {"train_samples": len(train_ds), "val_samples": len(val_ds)})
    if is_main(rank):
        console.print(
            f"[cyan]train={len(train_ds)}  val={len(val_ds)}  "
            f"batch/GPU={train_cfg.batch_size}  global_batch={train_cfg.batch_size * world_size}[/cyan]"
        )

    # ---- model ----
    model = GraspFlowModel(cfg, norm_stats=norm_stats).to(device)
    logger.info("model initialized", extra=log_context)
    if writer:
        writer.write("model_initialized")
    if train_cfg.get("pretrained"):
        # finetune init: weights only, step/optimizer/best_val start fresh
        load_pretrained(model, str(train_cfg.pretrained), device)
        logger.info("pretrained loaded: %s", train_cfg.pretrained, extra=log_context)
        if writer:
            writer.write("pretrained_loaded", {"path": str(train_cfg.pretrained)})
    model.train()
    ddp_model = (
        DDP(model, device_ids=[local_rank])
        if world_size > 1 and device.type == "cuda"
        else model
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=train_cfg.lr,
        betas=tuple(train_cfg.betas),
        weight_decay=train_cfg.weight_decay,
    )

    ema_model = AveragedModel(
        model, multi_avg_fn=get_ema_multi_avg_fn(train_cfg.ema_decay), use_buffers=False
    )

    # ---- resume ----
    start_step = 0
    best_val = float("inf")
    if train_cfg.get("resume"):
        ckpt = torch.load(train_cfg.resume, map_location=device, weights_only=False)
        # strict=False: checkpoints no longer store the frozen DINOv2 encoder
        model.load_state_dict(ckpt["model"], strict=False)
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("ema") is not None:
            ema_model.load_state_dict(ckpt["ema"], strict=False)
        start_step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_val") or float("inf"))
        if ckpt.get("val_metric") != VAL_METRIC:
            # Old metric semantics (x0-recovery proxy ~2x optimistic): the
            # stored best_val is not comparable -> reset and re-baseline.
            logger.info(
                f"resume: val_metric {ckpt.get('val_metric')!r} != {VAL_METRIC!r} "
                f"-> resetting best_val (was {best_val:.4f})",
                extra=log_context,
            )
            best_val = float("inf")
        if is_main(rank):
            console.print(
                f"[yellow]resumed from {train_cfg.resume} @ step {start_step} "
                f"(best_val={best_val:.4f})[/yellow]"
            )

    def lr_at(step: int) -> float:
        """Linear warmup -> cosine decay to lr * lr_min_ratio."""
        w = max(int(train_cfg.warmup_steps), 1)
        total = int(train_cfg.total_steps)
        min_ratio = float(train_cfg.get("lr_min_ratio", 0.0))
        if step < w:
            return train_cfg.lr * (step + 1) / w
        progress = min(max((step - w) / max(total - w, 1), 0.0), 1.0)
        return train_cfg.lr * (
            min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        )

    # ---- train loop ----
    bf16 = bool(train_cfg.bf16) and device.type == "cuda"
    loader_iter = infinite_loader(train_loader, sampler)
    heartbeat_every = max(int(train_cfg.get("heartbeat_every", train_cfg.log_every)), 1)
    t_start = time.perf_counter()
    last_log_t, last_log_step = t_start, start_step
    _safe_event(
        writer,
        logger,
        "training_started",
        {"total_steps": int(train_cfg.total_steps), "start_step": start_step},
        start_step,
    )
    logger.info(
        "training loop started: step=%d total=%d",
        start_step,
        int(train_cfg.total_steps),
        extra=log_context,
    )

    for step in range(start_step, int(train_cfg.total_steps)):
        lr = lr_at(step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        step_no = step + 1
        exception_state["step"] = step_no
        exception_state["phase"] = "batch"
        if step_no % heartbeat_every == 0:
            _safe_event(writer, logger, "heartbeat", {"phase": "batch"}, step_no)
        batch = next(loader_iter)
        exception_state["phase"] = "forward"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            preds, targets, time_weight = ddp_model(
                point_uv=batch["point_uv"].to(device, non_blocking=True),
                camera_K=batch["camera_K"].to(device, non_blocking=True),
                gt_mano_params=batch["mano_params"].to(device, non_blocking=True),
                rgb=batch["rgb"].to(device, non_blocking=True) if "rgb" in batch else None,
                pcl_xyz=(
                    batch["pcl_xyz"].to(device, non_blocking=True)
                    if "pcl_xyz" in batch
                    else None
                ),
                pcl_rgb=(
                    batch["pcl_rgb"].to(device, non_blocking=True)
                    if "pcl_rgb" in batch
                    else None
                ),
            )
        targets["landmarks_2d"] = batch["landmarks_2d"].to(device, non_blocking=True)
        targets["camera_K"] = batch["camera_K"].to(device, non_blocking=True)
        targets["image_size"] = int(cfg.trainer.model.get("image_size", 224))
        loss, comps = compute_loss(
            preds,
            targets,
            time_weight,
            train_cfg.lambda_v,
            train_cfg.lambda_3d,
            float(train_cfg.get("lambda_2d", 0.0)),
        )

        exception_state["phase"] = "backward"
        loss.backward()
        if train_cfg.grad_clip and train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), train_cfg.grad_clip
            )
        optimizer.step()
        if step >= int(train_cfg.ema_start_step):
            ema_model.update_parameters(model)

        # ---- logging ----
        if is_main(rank) and step_no % int(train_cfg.log_every) == 0:
            now = time.perf_counter()
            sps = (step_no - last_log_step) / max(now - last_log_t, 1e-9)
            last_log_t, last_log_step = now, step_no
            comps.update({"step": step_no, "lr": lr, "samples_per_s": sps})
            l2d_str = f"l2d {comps['l2d_px']:.2f}px  " if "l2d_px" in comps else ""
            console.print(
                f"step {step_no:>6d}  loss {comps['loss']:.4f}  "
                f"lv {comps['lv']:.4f}  l3d {comps['l3d']:.4f}  {l2d_str}"
                f"lr {lr:.2e}  {sps:.1f} it/s"
            )
            _safe_event(writer, logger, "train", comps, step_no)

        # ---- validation ----
        # best checkpoint score = mean over datasets of mean(PA-MPJPE, PA-MPVPE)
        # (mm), real sampling metrics. When EMA is active we validate the EMA
        # weights (what gets deployed/evaluated at test time), else the raw model.
        # All ranks evaluate their shard (rank-0-only would exceed the NCCL
        # watchdog timeout while other ranks wait at the barrier).
        if (step + 1) % int(train_cfg.val_every) == 0 or (step + 1) == int(
            train_cfg.total_steps
        ):
            ema_active = step + 1 >= int(train_cfg.ema_start_step)
            eval_model = ema_model.module if ema_active else model
            _phase_log(log_context, "validation", "start step=%d", step_no)
            _safe_event(writer, logger, "validation_started", {"weights": "ema" if ema_active else "model"}, step_no)
            exception_state["phase"] = "validation"
            val_results = run_val(eval_model, val_loaders, device, bf16, rank, world_size)
            exception_state["phase"] = "batch"
            _phase_log(log_context, "validation", "finished step=%d", step_no)
            ds_scores = {
                name: 0.5 * (r["pa_mpjpe"] + r["pa_mpvpe"])
                for name, r in val_results.items()
            }
            score = sum(ds_scores.values()) / len(ds_scores)
            if is_main(rank):
                console.print(f"[green]val @ {step + 1} ({'ema' if ema_active else 'model'}):[/green]")
                for name, r in val_results.items():
                    console.print(
                        f"  \\[{name}] MPJPE {r['mpjpe']:.2f} PA-MPJPE {r['pa_mpjpe']:.2f} "
                        f"MPVPE {r['mpvpe']:.2f} PA-MPVPE {r['pa_mpvpe']:.2f} mm"
                    )
                console.print(f"  [bold green]score {score:.2f}[/bold green]")
            _safe_event(
                writer,
                logger,
                "validation",
                {
                    "val": True,
                    "weights": "ema" if ema_active else "model",
                    "datasets": val_results,
                    "score": score,
                },
                step_no,
            )
            if is_main(rank) and score < best_val:
                best_val = score
                ckpt_path = out_dir / "model_best.pt"
                exception_state["phase"] = "checkpoint_best"
                _safe_event(
                    writer,
                    logger,
                    "checkpoint_started",
                    {"path": str(ckpt_path), "kind": "best"},
                    step_no,
                )
                try:
                    save_checkpoint(
                        ckpt_path,
                        model,
                        ema_model,
                        optimizer,
                        cfg,
                        norm_stats,
                        step_no,
                        best_val=best_val,
                    )
                except BaseException:
                    logger.exception(
                        "checkpoint failed: path=%s step=%d",
                        ckpt_path,
                        step_no,
                        extra=log_context,
                    )
                    raise
                _safe_event(
                    writer,
                    logger,
                    "checkpoint_finished",
                    {"path": str(ckpt_path), "kind": "best"},
                    step_no,
                )
                console.print(
                    f"[bold green]new best score {best_val:.4f} -> "
                    f"saved model_best.pt @ step {step_no}[/bold green]"
                )
                exception_state["phase"] = "batch"
            if world_size > 1:
                dist.barrier()

        if (step + 1) % int(train_cfg.ckpt_every) == 0 or (step + 1) == int(
            train_cfg.total_steps
        ):
            if is_main(rank):
                ckpt_path = out_dir / "model.pt"
                exception_state["phase"] = "checkpoint_periodic"
                _safe_event(
                    writer,
                    logger,
                    "checkpoint_started",
                    {"path": str(ckpt_path), "kind": "periodic"},
                    step_no,
                )
                try:
                    save_checkpoint(
                        ckpt_path,
                        model,
                        ema_model,
                        optimizer,
                        cfg,
                        norm_stats,
                        step_no,
                        best_val=best_val,
                    )
                except BaseException:
                    logger.exception(
                        "checkpoint failed: path=%s step=%d",
                        ckpt_path,
                        step_no,
                        extra=log_context,
                    )
                    raise
                _safe_event(
                    writer,
                    logger,
                    "checkpoint_finished",
                    {"path": str(ckpt_path), "kind": "periodic"},
                    step_no,
                )
                console.print(f"[magenta]saved checkpoint @ step {step_no}[/magenta]")
                exception_state["phase"] = "batch"
            if world_size > 1:
                _phase_log(log_context, "barrier", "enter periodic checkpoint step=%d", step_no)
                dist.barrier()
                _phase_log(log_context, "barrier", "exit periodic checkpoint step=%d", step_no)

    _safe_event(writer, logger, "run_finished", {"status": "success"}, int(train_cfg.total_steps))
    logger.info("training finished successfully", extra=log_context)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        tyro.cli(main)
    except BaseException:
        # The installed hooks normally record this with the full traceback.
        # This fallback also guarantees stderr visibility if logging setup
        # itself failed before handlers could be installed.
        logger.exception("training process terminated", extra={"rank": os.environ.get("RANK", "0")})
        logging.shutdown()
        raise
