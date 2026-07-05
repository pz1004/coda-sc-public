"""Real multi-exit CNN experiments for CODA-SC (revision path 1a).

This module replaces the earlier sklearn/toy setup with a genuine, GPU-trained
multi-exit convolutional network on real vision workloads.  Crucially, the
model-internal quantities used by the policy layer -- activation payload size
and per-stage FLOPs -- are measured from the trained network rather than
hand-set.  Latency and energy are then derived from documented device
assumptions.

Pipeline per (dataset, seed):
  1. train a small VGG-style backbone with three early-exit heads,
  2. measure int8 activation payloads and per-stage FLOPs for every split point
     and exit,
  3. export calibration/test exit probabilities and full-model probabilities.

The exported probability bundles + measured profile are consumed by the
conformal-exit and online-control policies (see ``policy_eval.py``), so the
policy sweep runs on cached numpy arrays without re-invoking the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Edge/cloud device model (FLOPs-based, in the spirit of Neurosurgeon and
# Auto-Split, which drive placement from per-layer latency/energy models). ---
# Compute latency and energy are derived from measured per-stage FLOPs and
# documented device throughput/efficiency -- NOT hand-set per-action numbers.
# Defaults emulate a constrained edge CPU (Raspberry-Pi-4 / Cortex-A72 class)
# and a server-class GPU; EDGE_GFLOPS is treated as a modelling assumption and
# is swept in the sensitivity analysis.
EDGE_GFLOPS = 5.0            # effective edge-CPU throughput (GFLOP/s)
CLOUD_GFLOPS = 5000.0        # effective server-GPU throughput (GFLOP/s)
EDGE_J_PER_GFLOP = 0.10      # edge compute energy (~10 GFLOP/J mobile SoC)
UPLOAD_J_PER_KB = 5.0e-4     # wireless upload energy per KB
BYTES_PER_ACT = 1           # int8-quantised activations for transmission

DATASETS: dict[str, dict] = {
    # Deadlines are application SLOs chosen so that early exits comfortably meet
    # them, offloading meets them under good networks but not congested ones, and
    # full local inference is near the deadline (sometimes meeting, sometimes
    # missing) -- creating a genuine exit/offload/local trade-off.
    "mnist": dict(display="MNIST", ch=1, classes=10, epochs=8, deadline_ms=70.0),
    "fashion_mnist": dict(display="Fashion-MNIST", ch=1, classes=10, epochs=16, deadline_ms=75.0),
    "cifar10": dict(display="CIFAR-10", ch=3, classes=10, epochs=32, deadline_ms=75.0),
    "svhn": dict(display="SVHN", ch=3, classes=10, epochs=22, deadline_ms=80.0),
}

CAL_SIZE = 5000            # held-out calibration split carved from the train set
WIDTHS = (64, 128, 256)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _transform(ch: int) -> transforms.Compose:
    mean = (0.5,) * ch
    std = (0.5,) * ch
    return transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean, std)]
    )


def load_dataset(name: str, root: str):
    tfm = _transform(DATASETS[name]["ch"])
    if name == "mnist":
        tr = torchvision.datasets.MNIST(root, train=True, download=True, transform=tfm)
        te = torchvision.datasets.MNIST(root, train=False, download=True, transform=tfm)
    elif name == "fashion_mnist":
        tr = torchvision.datasets.FashionMNIST(root, train=True, download=True, transform=tfm)
        te = torchvision.datasets.FashionMNIST(root, train=False, download=True, transform=tfm)
    elif name == "cifar10":
        tr = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tfm)
        te = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=tfm)
    elif name == "svhn":
        tr = torchvision.datasets.SVHN(root, split="train", download=True, transform=tfm)
        te = torchvision.datasets.SVHN(root, split="test", download=True, transform=tfm)
    else:
        raise ValueError(f"unknown dataset: {name}")
    return tr, te


def _split_train_cal(train_set, seed: int):
    n = len(train_set)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cal_idx = perm[:CAL_SIZE].tolist()
    tr_idx = perm[CAL_SIZE:].tolist()
    return Subset(train_set, tr_idx), Subset(train_set, cal_idx)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.net(x)


class ExitHead(nn.Module):
    def __init__(self, cin: int, classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cin, 3, padding=1),
            nn.BatchNorm2d(cin),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cin, classes),
        )

    def forward(self, x):
        return self.net(x)


class MultiExitCNN(nn.Module):
    """Small VGG-style backbone with three early-exit heads and a final head.

    Split point ``k`` means the first ``k`` blocks run on the edge and the
    remaining ``3-k`` blocks plus the final head run in the cloud (``k=0`` sends
    the raw input, ``k=3`` sends the deepest activation).  Exit ``e`` returns the
    prediction of the ``e``-th early head after ``e`` local blocks.
    """

    def __init__(self, ch: int, classes: int, widths=WIDTHS):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ConvBlock(ch, widths[0]), ConvBlock(widths[0], widths[1]), ConvBlock(widths[1], widths[2])]
        )
        self.exits = nn.ModuleList([ExitHead(w, classes) for w in widths])
        self.final = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(widths[2], classes))

    def forward(self, x):
        logits_exits = []
        h = x
        for i, block in enumerate(self.blocks):
            h = block(h)
            logits_exits.append(self.exits[i](h))
        return logits_exits, self.final(h)

    # -- sub-forward passes used for measurement and offload prediction -- #
    def prefix(self, x, k: int):
        for i in range(k):
            x = self.blocks[i](x)
        return x

    def suffix(self, h, k: int):
        for i in range(k, len(self.blocks)):
            h = self.blocks[i](h)
        return self.final(h)

    def exit_logits(self, x, e: int):
        h = self.prefix(x, e)
        return self.exits[e - 1](h)

    def full_logits(self, x):
        return self.suffix(x, 0)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def make_loaders(name: str, seed: int, root: str, train_batch: int = 128,
                 eval_batch: int = 512) -> tuple[DataLoader, DataLoader, DataLoader]:
    cfg = DATASETS[name]
    train_set, test_set = load_dataset(name, root)
    tr_subset, cal_subset = _split_train_cal(train_set, seed)

    tr_loader = DataLoader(tr_subset, batch_size=train_batch, shuffle=True, num_workers=4, drop_last=True)
    cal_loader = DataLoader(cal_subset, batch_size=eval_batch, shuffle=False, num_workers=4)
    te_loader = DataLoader(test_set, batch_size=eval_batch, shuffle=False, num_workers=4)
    return tr_loader, cal_loader, te_loader


def train_model(name: str, seed: int, root: str) -> tuple[MultiExitCNN, DataLoader, DataLoader, DataLoader]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = DATASETS[name]
    tr_loader, cal_loader, te_loader = make_loaders(name, seed, root)

    model = MultiExitCNN(cfg["ch"], cfg["classes"]).to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    exit_w = [0.5, 0.5, 0.5]

    model.train()
    for epoch in range(cfg["epochs"]):
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits_exits, logits_final = model(xb)
            loss = F.cross_entropy(logits_final, yb)
            for w, le in zip(exit_w, logits_exits):
                loss = loss + w * F.cross_entropy(le, yb)
            loss.backward()
            opt.step()
        sched.step()
    return model, tr_loader, cal_loader, te_loader


def load_model_from_checkpoint(name: str, checkpoint_path: Path) -> MultiExitCNN:
    cfg = DATASETS[name]
    model = MultiExitCNN(cfg["ch"], cfg["classes"]).to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def export_probs(model: MultiExitCNN, loader: DataLoader) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Return (exit_probs[3], full_probs, labels) as numpy arrays."""
    model.eval()
    exit_p = [[], [], []]
    full_p = []
    ys = []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        logits_exits, logits_final = model(xb)
        for e in range(3):
            exit_p[e].append(F.softmax(logits_exits[e], dim=1).cpu().numpy())
        full_p.append(F.softmax(logits_final, dim=1).cpu().numpy())
        ys.append(yb.numpy())
    exit_probs = [np.concatenate(p).astype(np.float64) for p in exit_p]
    return exit_probs, np.concatenate(full_p).astype(np.float64), np.concatenate(ys).astype(int)


# --------------------------------------------------------------------------- #
# Measurement (FLOPs -> latency/energy, payloads)  -- the anti-fabrication core
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure_flops(model: MultiExitCNN, sample: torch.Tensor) -> tuple[list[float], list[float], float]:
    """Per-sample FLOPs of each backbone block, each exit head, and the final
    head, obtained from real tensor shapes via forward hooks (Conv/Linear/BN)."""
    store: dict[int, float] = {}

    def make_hook(m):
        def hook(mod, inp, out):
            if isinstance(mod, nn.Conv2d):
                cout, hout, wout = out.shape[1], out.shape[2], out.shape[3]
                f = 2.0 * cout * hout * wout * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
                if mod.bias is not None:
                    f += cout * hout * wout
            elif isinstance(mod, nn.Linear):
                f = 2.0 * mod.in_features * mod.out_features + (mod.out_features if mod.bias is not None else 0)
            elif isinstance(mod, nn.BatchNorm2d):
                f = 2.0 * out.shape[1] * out.shape[2] * out.shape[3]
            else:
                return
            store[id(mod)] = store.get(id(mod), 0.0) + f
        return hook

    handles = [m.register_forward_hook(make_hook(m))
               for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d))]
    model.eval()
    model(sample[:1])  # single-sample forward runs every block, exit head, and final head once
    for h in handles:
        h.remove()

    def tree(mod) -> float:
        return sum(store.get(id(m), 0.0) for m in mod.modules())

    block_flops = [tree(b) for b in model.blocks]
    exit_flops = [tree(e) for e in model.exits]
    final_flops = tree(model.final)
    return block_flops, exit_flops, final_flops


@torch.no_grad()
def measure_profile(model: MultiExitCNN, name: str, sample: torch.Tensor) -> dict:
    """Assemble the measured systems profile: per-split payload (int8 activation
    bytes), and compute latency/energy derived from measured FLOPs and the
    documented edge/cloud device model.  Nothing here is hand-set per action."""
    cfg = DATASETS[name]
    dev = next(model.parameters()).device
    sample = sample.to(dev)
    block_flops, exit_flops, final_flops = measure_flops(model, sample)

    def edge_ms(flops):
        return flops / 1e9 / EDGE_GFLOPS * 1000.0

    def cloud_ms(flops):
        return flops / 1e9 / CLOUD_GFLOPS * 1000.0

    def edge_j(flops):
        return flops / 1e9 * EDGE_J_PER_GFLOP

    # payload = exact int8 activation tensor size at each split point
    payload_kb = {k: model.prefix(sample[:1], k).numel() * BYTES_PER_ACT / 1024.0 for k in range(4)}
    prefix_flops = [sum(block_flops[:k]) for k in range(4)]              # k local blocks
    suffix_flops = [sum(block_flops[k:]) + final_flops for k in range(4)]  # cloud runs the rest
    full_flops = sum(block_flops) + final_flops

    names = {0: "cloud_raw", 1: "split_l1", 2: "split_l2", 3: "split_l3"}
    actions = [
        dict(
            name=names[k],
            split=k,
            local_ms=round(edge_ms(prefix_flops[k]), 4),
            cloud_ms=round(cloud_ms(suffix_flops[k]), 4),
            payload_kb=round(payload_kb[k], 4),
            local_j=round(edge_j(prefix_flops[k]), 6),
            local_gflops=round(prefix_flops[k] / 1e9, 5),
        )
        for k in range(4)
    ]
    exits = {}
    for e in (1, 2, 3):
        ef = sum(block_flops[:e]) + exit_flops[e - 1]
        exits[e] = dict(local_ms=round(edge_ms(ef), 4), local_j=round(edge_j(ef), 6),
                        local_gflops=round(ef / 1e9, 5))

    return dict(
        dataset=name,
        display=cfg["display"],
        deadline_ms=cfg["deadline_ms"],
        local_full_ms=round(edge_ms(full_flops), 4),
        local_full_j=round(edge_j(full_flops), 6),
        local_full_gflops=round(full_flops / 1e9, 5),
        raw_input_kb=round(int(sample[:1].numel()) * BYTES_PER_ACT / 1024.0, 4),
        edge_gflops=EDGE_GFLOPS,
        cloud_gflops=CLOUD_GFLOPS,
        edge_j_per_gflop=EDGE_J_PER_GFLOP,
        upload_j_per_kb=UPLOAD_J_PER_KB,
        bytes_per_act=BYTES_PER_ACT,
        actions=actions,
        exits=exits,
    )


# --------------------------------------------------------------------------- #
# Learned split bottlenecks
# --------------------------------------------------------------------------- #
class SplitBottleneck(nn.Module):
    """Tiny 1x1-convolution codec trained on a frozen backbone activation."""

    def __init__(self, channels: int, bottleneck_channels: int):
        super().__init__()
        self.encoder = nn.Conv2d(channels, bottleneck_channels, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.decoder = nn.Conv2d(bottleneck_channels, channels, kernel_size=1)

    def encode(self, h):
        return self.act(self.encoder(h))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, h):
        return self.decode(self.encode(h))


def _conv1x1_flops(cin: int, cout: int, h: int, w: int, bias: bool = True) -> float:
    flops = 2.0 * cin * cout * h * w
    if bias:
        flops += cout * h * w
    return flops


def _edge_ms(flops: float) -> float:
    return flops / 1e9 / EDGE_GFLOPS * 1000.0


def _cloud_ms(flops: float) -> float:
    return flops / 1e9 / CLOUD_GFLOPS * 1000.0


def _edge_j(flops: float) -> float:
    return flops / 1e9 * EDGE_J_PER_GFLOP


def _codec_path(cache_dir: Path, dataset: str, seed: int, split: int, channels: int) -> Path:
    return cache_dir / f"{dataset}_s{seed}_bottleneck_l{split}_c{channels}.pt"


def _bottleneck_npz_path(cache_dir: Path, dataset: str, seed: int, channels: int) -> Path:
    return cache_dir / f"{dataset}_s{seed}_bottleneck_c{channels}.npz"


def _model_checkpoint_path(cache_dir: Path, dataset: str, seed: int) -> Path:
    return cache_dir / f"{dataset}_s{seed}.pt"


def train_bottleneck(model: MultiExitCNN, loader: DataLoader, split: int,
                     bottleneck_channels: int, epochs: int, path: Path) -> SplitBottleneck:
    sample_x, _ = next(iter(loader))
    with torch.no_grad():
        h = model.prefix(sample_x[:1].to(DEVICE), split)
    codec = SplitBottleneck(h.shape[1], bottleneck_channels).to(DEVICE)
    opt = torch.optim.Adam(codec.parameters(), lr=1e-3)

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    codec.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with torch.no_grad():
                target_h = model.prefix(xb, split)
            decoded = codec(target_h)
            logits = model.suffix(decoded, split)
            loss = F.cross_entropy(logits, yb) + 0.1 * F.mse_loss(decoded, target_h.detach())
            opt.zero_grad()
            loss.backward()
            opt.step()
    torch.save({"state_dict": codec.state_dict(), "split": split,
                "bottleneck_channels": bottleneck_channels}, path)
    return codec


def load_or_train_bottlenecks(model: MultiExitCNN, train_loader: DataLoader, cache_dir: Path,
                              dataset: str, seed: int, bottleneck_channels: int,
                              epochs: int) -> dict[int, SplitBottleneck]:
    codecs = {}
    for split in (1, 2, 3):
        sample_x, _ = next(iter(train_loader))
        with torch.no_grad():
            channels = model.prefix(sample_x[:1].to(DEVICE), split).shape[1]
        path = _codec_path(cache_dir, dataset, seed, split, bottleneck_channels)
        codec = SplitBottleneck(channels, bottleneck_channels).to(DEVICE)
        if path.exists():
            state = torch.load(path, map_location=DEVICE)
            codec.load_state_dict(state["state_dict"] if "state_dict" in state else state)
        else:
            codec = train_bottleneck(model, train_loader, split, bottleneck_channels, epochs, path)
        codec.eval()
        codecs[split] = codec
    return codecs


@torch.no_grad()
def export_bottleneck_probs(model: MultiExitCNN, codecs: dict[int, SplitBottleneck],
                            loader: DataLoader) -> tuple[dict[int, np.ndarray], np.ndarray]:
    model.eval()
    for codec in codecs.values():
        codec.eval()
    split_p = {k: [] for k in codecs}
    ys = []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        for split, codec in codecs.items():
            h = model.prefix(xb, split)
            logits = model.suffix(codec(h), split)
            split_p[split].append(F.softmax(logits, dim=1).cpu().numpy())
        ys.append(yb.numpy())
    return {k: np.concatenate(v).astype(np.float64) for k, v in split_p.items()}, np.concatenate(ys).astype(int)


@torch.no_grad()
def bottleneck_profile(model: MultiExitCNN, base_profile: dict, sample: torch.Tensor,
                       bottleneck_channels: int) -> dict:
    """Measured profile for learned 1x1 codecs. Raw input remains unchanged; split
    actions transmit the encoded int8 tensor and include encoder/decoder FLOPs."""
    import copy

    p = copy.deepcopy(base_profile)
    sample = sample.to(DEVICE)
    block_flops, _, final_flops = measure_flops(model, sample)
    prefix_flops = [sum(block_flops[:k]) for k in range(4)]
    suffix_flops = [sum(block_flops[k:]) + final_flops for k in range(4)]
    action_by_split = {a["split"]: a for a in p["actions"]}
    for split in (1, 2, 3):
        h = model.prefix(sample[:1], split)
        _, c, ht, wt = h.shape
        enc_flops = _conv1x1_flops(c, bottleneck_channels, ht, wt)
        dec_flops = _conv1x1_flops(bottleneck_channels, c, ht, wt)
        action_by_split[split]["local_ms"] = round(_edge_ms(prefix_flops[split] + enc_flops), 4)
        action_by_split[split]["cloud_ms"] = round(_cloud_ms(dec_flops + suffix_flops[split]), 4)
        action_by_split[split]["payload_kb"] = round(bottleneck_channels * ht * wt * BYTES_PER_ACT / 1024.0, 4)
        action_by_split[split]["local_j"] = round(_edge_j(prefix_flops[split] + enc_flops), 6)
        action_by_split[split]["local_gflops"] = round((prefix_flops[split] + enc_flops) / 1e9, 5)
        action_by_split[split]["bottleneck_channels"] = bottleneck_channels
    p["bottleneck_channels"] = bottleneck_channels
    return p


# --------------------------------------------------------------------------- #
# Orchestration: train, measure, export, cache
# --------------------------------------------------------------------------- #
def train_measure_export(name: str, seed: int, data_root: str, cache_dir: Path) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / f"{name}_s{seed}.npz"
    ckpt_path = _model_checkpoint_path(cache_dir, name, seed)

    model, _, cal_loader, te_loader = train_model(name, seed, data_root)
    exit_cal, full_cal, y_cal = export_probs(model, cal_loader)
    exit_test, full_test, y_test = export_probs(model, te_loader)

    test_acc = float(np.mean(full_test.argmax(1) == y_test))

    save = {"y_cal": y_cal, "y_test": y_test, "full_cal": full_cal, "full_test": full_test}
    for e in range(3):
        save[f"exit_cal_{e}"] = exit_cal[e]
        save[f"exit_test_{e}"] = exit_test[e]
    np.savez_compressed(npz_path, **save)
    torch.save({"state_dict": model.state_dict(), "dataset": name, "seed": seed}, ckpt_path)

    # measure the profile once per dataset (architecture/hardware dependent, not seed)
    sample, _ = next(iter(te_loader))
    profile = measure_profile(model, name, sample[:256])
    profile["test_accuracy"] = round(test_acc, 4)
    return profile


def export_learned_bottlenecks(name: str, seed: int, data_root: str, cache_dir: Path,
                               base_profile: dict, bottleneck_channels: int = 8,
                               epochs: int = 8) -> dict:
    """Train/load frozen-backbone codecs and export action-specific split logits."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = _model_checkpoint_path(cache_dir, name, seed)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"missing CNN checkpoint {ckpt_path}; rerun training so learned bottlenecks can use frozen weights"
        )
    model = load_model_from_checkpoint(name, ckpt_path)
    train_loader, cal_loader, te_loader = make_loaders(name, seed, data_root, train_batch=256, eval_batch=512)
    codecs = load_or_train_bottlenecks(model, train_loader, cache_dir, name, seed,
                                       bottleneck_channels, epochs)
    split_cal, y_cal = export_bottleneck_probs(model, codecs, cal_loader)
    split_test, y_test = export_bottleneck_probs(model, codecs, te_loader)

    save = {"y_cal": y_cal, "y_test": y_test}
    for split in (1, 2, 3):
        save[f"split_cal_{split}"] = split_cal[split]
        save[f"split_test_{split}"] = split_test[split]
    np.savez_compressed(_bottleneck_npz_path(cache_dir, name, seed, bottleneck_channels), **save)

    sample, _ = next(iter(te_loader))
    return bottleneck_profile(model, base_profile, sample[:256], bottleneck_channels)


def run_training(seeds: list[int], data_root: str, cache_dir: Path, profiles_path: Path,
                 datasets: list[str] | None = None) -> dict:
    datasets = datasets or list(DATASETS.keys())
    profiles: dict[str, dict] = {}
    accs: dict[str, list[float]] = {}
    for name in datasets:
        for seed in seeds:
            prof = train_measure_export(name, seed, data_root, cache_dir)
            accs.setdefault(name, []).append(prof["test_accuracy"])
            if name not in profiles:  # keep first-seed measured profile
                profiles[name] = prof
            print(f"[{name} seed={seed}] test_acc={prof['test_accuracy']:.4f}", flush=True)
    for name in datasets:
        profiles[name]["test_accuracy_mean"] = round(float(np.mean(accs[name])), 4)
        profiles[name]["test_accuracy_std"] = round(float(np.std(accs[name])), 4)
        profiles[name]["seeds"] = list(seeds)
    profiles_path.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    return profiles


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7")
    ap.add_argument("--datasets", default="")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--cache", default="results/cnn_cache")
    ap.add_argument("--profiles", default="results/profiles.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s]
    ds = [d for d in args.datasets.split(",") if d] or None
    run_training(seeds, args.data_root, Path(args.cache), Path(args.profiles), ds)
