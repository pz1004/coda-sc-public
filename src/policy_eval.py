"""Policy evaluation on cached CNN logits + measured systems profiles.

Given a trained multi-exit CNN's exported probabilities (see ``coda_cnn.py``) and
its measured FLOPs-based device profile, this module runs every inference policy
-- baselines, ablations, reference policies, and CODA-SC -- over Internet-like
network traces and returns per-run metrics.

The CODA-SC controller here is *coupled*: for uncertain samples the same
deadline dual variable that tracks the miss budget chooses between offloading
and a local full-model fallback.  Coverage-mode CODA-SC uses split-conformal
singleton exits; risk-mode CODA-SC uses CRC-calibrated singleton exits.
The optional ``coda_controller`` and ``coda_split_controller`` policies disable
exits and are used only for split-selection stress tests that isolate the
controller action space with and without the local fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from coda_sc import SplitAction, action_energy, action_latency, generate_trace

PROFILES = ["wifi", "lte", "congested", "mixed"]
PROFILE_OFFSETS = {"wifi": 101, "lte": 211, "congested": 307, "mixed": 419}
DATASET_OFFSETS = {"mnist": 1000, "fashion_mnist": 2000, "cifar10": 3000, "svhn": 4000}

# controller weights (deadline dominates; energy/upload are secondary objectives)
BETA_ENERGY = 6.0
GAMMA_UPLOAD = 0.05
ETA_DUAL = 0.03

ACTION_RATE_KEYS = ("cloud_raw", "split_l1", "split_l2", "split_l3", "local_full")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_bundle(cache_dir: Path, dataset: str, seed: int, bottleneck_channels: int | None = None) -> dict:
    d = np.load(cache_dir / f"{dataset}_s{seed}.npz")
    out = {
        "y_cal": d["y_cal"],
        "y_test": d["y_test"],
        "full_cal": d["full_cal"],
        "full_test": d["full_test"],
        "exit_cal": [d[f"exit_cal_{e}"] for e in range(3)],
        "exit_test": [d[f"exit_test_{e}"] for e in range(3)],
    }
    if bottleneck_channels is not None:
        b = np.load(cache_dir / f"{dataset}_s{seed}_bottleneck_c{bottleneck_channels}.npz")
        if not np.array_equal(out["y_cal"], b["y_cal"]) or not np.array_equal(out["y_test"], b["y_test"]):
            raise ValueError(f"bottleneck labels do not match base cache for {dataset} seed {seed}")
        out["split_test_by_name"] = {
            "split_l1": b["split_test_1"],
            "split_l2": b["split_test_2"],
            "split_l3": b["split_test_3"],
        }
    return out


def rescale_profile(profile: dict, edge_gflops: float) -> dict:
    """Return a copy of the measured profile with edge compute latency/energy
    recomputed for a different edge throughput (used by the sensitivity sweep).
    FLOPs are fixed measurements; only the device throughput assumption changes."""
    import copy

    p = copy.deepcopy(profile)
    ref = profile["edge_gflops"]
    scale = ref / edge_gflops
    for a in p["actions"]:
        a["local_ms"] = round(a["local_ms"] * scale, 4)
        a["local_j"] = round(a["local_j"], 6)  # energy is FLOP-proportional, throughput-independent
    for e in p["exits"].values():
        e["local_ms"] = round(e["local_ms"] * scale, 4)
    p["local_full_ms"] = round(profile["local_full_ms"] * scale, 4)
    p["edge_gflops"] = edge_gflops
    return p


def rescale_payloads(profile: dict, payload_scale: float, raw_input_scale: float = 1.0) -> dict:
    """Return a profile whose transmitted activations are compressed by a fixed
    factor.  Raw input upload can also be inflated to emulate larger input
    frames.  This is a no-hardware diagnostic for when split-point selection
    matters."""
    import copy

    p = copy.deepcopy(profile)
    for a in p["actions"]:
        if a["split"] == 0:
            a["payload_kb"] = round(a["payload_kb"] * raw_input_scale, 4)
        else:
            a["payload_kb"] = round(a["payload_kb"] * payload_scale, 4)
    p["payload_scale"] = payload_scale
    p["raw_input_scale"] = raw_input_scale
    return p


def actions_from_profile(profile: dict) -> tuple[list[SplitAction], SplitAction, dict, float]:
    up = profile["upload_j_per_kb"]
    actions = [
        SplitAction(
            name=a["name"], split=a["split"], local_ms=a["local_ms"], cloud_ms=a["cloud_ms"],
            payload_kb=a["payload_kb"], local_j=a["local_j"], upload_j_per_kb=up,
        )
        for a in profile["actions"]
    ]
    local_full = SplitAction("local_full", 4, profile["local_full_ms"], 0.0, 0.0, profile["local_full_j"], up)
    exits = {int(k): v for k, v in profile["exits"].items()}
    return actions, local_full, exits, float(profile["deadline_ms"])


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def conformal_taus(exit_cal: list[np.ndarray], y_cal: np.ndarray, alpha: float) -> list[float]:
    taus = []
    n = len(y_cal)
    rank = min(max(int(np.ceil((n + 1) * (1.0 - alpha))), 1), n)
    for probs in exit_cal:
        scores = 1.0 - probs[np.arange(n), y_cal]
        taus.append(float(np.partition(scores, rank - 1)[rank - 1]))
    return taus


def confidence_taus_matched(exit_cal: list[np.ndarray], y_cal: np.ndarray, target_sel_err: float) -> list[float]:
    """Per-exit confidence threshold at a *matched operating point*: the lowest
    threshold whose calibration selective error is <= target (so confidence-exit
    and conformal-exit are compared at the same reliability target, not arbitrary
    tuning)."""
    taus = []
    for probs in exit_cal:
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        chosen = 0.999
        for tau in np.linspace(0.5, 0.999, 200):
            mask = conf >= tau
            if mask.sum() < 10:
                continue
            sel_err = float(np.mean(pred[mask] != y_cal[mask]))
            if sel_err <= target_sel_err:
                chosen = float(tau)
                break
        taus.append(chosen)
    return taus


def _entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def entropy_taus_matched(exit_cal: list[np.ndarray], y_cal: np.ndarray, target_sel_err: float) -> list[float]:
    """BranchyNet-style entropy thresholds at a matched selective-error target:
    the largest entropy threshold whose calibration selective error is <= target."""
    taus = []
    for probs in exit_cal:
        ent = _entropy(probs)
        pred = probs.argmax(axis=1)
        chosen = 0.0
        for tau in np.linspace(float(ent.max()), 0.0, 200):
            mask = ent <= tau
            if mask.sum() < 10:
                continue
            if float(np.mean(pred[mask] != y_cal[mask])) <= target_sel_err:
                chosen = float(tau)
                break
        taus.append(chosen)
    return taus


def crc_taus(exit_cal: list[np.ndarray], y_cal: np.ndarray, target_sel_err: float) -> list[float]:
    """Conformal risk control style thresholds: pick, per exit, the score
    threshold whose calibration singleton selective error is controlled at the
    target level (directly targeting selective error rather than set coverage)."""
    taus = []
    n = len(y_cal)
    for probs in exit_cal:
        best = 0.0
        for q in np.linspace(0.0, 0.9, 181):
            in_set = (1.0 - probs) <= q
            sizes = in_set.sum(axis=1)
            singleton = sizes == 1
            if singleton.sum() < 10:
                continue
            pred = probs.argmax(axis=1)
            sel_err = float(np.mean(pred[singleton] != y_cal[singleton]))
            # inflate by finite-sample slack (Hoeffding-style, matches CRC spirit)
            slack = np.sqrt(np.log(1 / 0.1) / (2 * max(1, singleton.sum())))
            if sel_err + slack <= target_sel_err:
                best = float(q)
        taus.append(best)
    return taus


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
STREAM_PER_REPEAT = 2000  # bounded online request stream (independent of test-set size)


def _repeat_stream(n: int, repeats: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    length = repeats * STREAM_PER_REPEAT
    out, total = [], 0
    while total < length:
        perm = rng.permutation(n)
        out.append(perm)
        total += len(perm)
    return np.concatenate(out)[:length]


def _conformal_exit(exit_probs, i, taus):
    """Return (exit_layer, pred) for the first singleton conformal set."""
    for layer, (probs, tau) in enumerate(zip(exit_probs, taus), start=1):
        size = int(np.sum((1.0 - probs[i]) <= tau))
        if size == 1:
            return layer, int(np.argmax(probs[i]))
    return None, None


def _confidence_exit(exit_probs, i, taus):
    for layer, (probs, tau) in enumerate(zip(exit_probs, taus), start=1):
        if float(np.max(probs[i])) >= tau:
            return layer, int(np.argmax(probs[i]))
    return None, None


def _entropy_exit(exit_probs, i, taus):
    for layer, (probs, tau) in enumerate(zip(exit_probs, taus), start=1):
        p = np.clip(probs[i], 1e-12, 1.0)
        if float(-np.sum(p * np.log(p))) <= tau:
            return layer, int(np.argmax(probs[i]))
    return None, None


def evaluate_policy(method, bundle, order, trace, actions, local_full, exits, deadline,
                    conf_taus, cnf_taus, ent_taus, crc_t, static_action, miss_budget=0.08):
    exit_test = bundle["exit_test"]
    full_test = bundle["full_test"]
    split_test_by_name = bundle.get("split_test_by_name", {})
    y = bundle["y_test"]
    n = len(order)

    def exit_cost(layer):
        return exits[layer]["local_ms"], exits[layer]["local_j"]

    # EMA network estimates (bootstrapped from the trace head)
    bw = float(np.median(trace.bandwidth_mbps[:20]))
    rtt = float(np.median(trace.rtt_ms[:20]))
    q = float(np.median(trace.queue_ms[:20]))
    dual = 0.0

    def est_latency(a: SplitAction):
        if a.payload_kb <= 0:
            return a.local_ms
        return a.local_ms + a.cloud_ms + rtt + q + a.payload_kb * 8.0 / max(bw, 1e-6)

    def controller_choice(cands):
        def J(a):
            est = est_latency(a)
            return est + dual * max(0.0, est - deadline) + BETA_ENERGY * action_energy(a) + GAMMA_UPLOAD * a.payload_kb
        return min(cands, key=J)

    def pred_for_action(a: SplitAction, idx: int) -> int:
        probs = split_test_by_name.get(a.name, full_test)
        return int(np.argmax(probs[idx]))

    preds = np.empty(n, dtype=int)
    lat = np.empty(n)
    up = np.empty(n)
    en = np.empty(n)
    kind = np.empty(n, dtype=object)  # 'exit' | 'offload' | 'local'
    sel_err = np.zeros(n, dtype=bool)
    accepted = np.zeros(n, dtype=bool)
    action_counts = {name: 0 for name in ACTION_RATE_KEYS}

    def record_action(action: SplitAction | None) -> None:
        if action is None:
            return
        if action.name in action_counts:
            action_counts[action.name] += 1

    for t, i in enumerate(order):
        a_used = None
        if method == "local_only":
            preds[t], lat[t], up[t], en[t], kind[t] = int(np.argmax(full_test[i])), local_full.local_ms, 0.0, action_energy(local_full), "local"
            record_action(local_full)
        elif method in ("cloud_only", "static_split", "oracle_split", "deadline_greedy"):
            if method == "cloud_only":
                a_used = actions[0]
            elif method == "static_split":
                a_used = static_action
            elif method == "oracle_split":
                a_used = min(actions, key=lambda a: action_latency(a, trace, t))
            else:  # deadline_greedy
                feas = [a for a in actions if est_latency(a) <= deadline]
                a_used = min(feas, key=action_energy) if feas else min(actions, key=est_latency)
            preds[t] = pred_for_action(a_used, i)
            lat[t], up[t], en[t], kind[t] = action_latency(a_used, trace, t), a_used.payload_kb, action_energy(a_used), "offload"
            record_action(a_used)
        elif method in ("coda_controller", "coda_split_controller"):
            cands = actions + [local_full] if method == "coda_controller" else actions
            a_used = controller_choice(cands)
            preds[t] = pred_for_action(a_used, i)
            if a_used.payload_kb <= 0:
                lat[t], up[t], en[t], kind[t] = a_used.local_ms, 0.0, action_energy(a_used), "local"
            else:
                lat[t], up[t], en[t], kind[t] = action_latency(a_used, trace, t), a_used.payload_kb, action_energy(a_used), "offload"
            record_action(a_used)
        elif method in ("confidence_exit", "branchynet", "conformal_exit", "crc_exit", "coda_sc", "coda_risk"):
            if method == "confidence_exit":
                layer, pe = _confidence_exit(exit_test, i, cnf_taus)
            elif method == "branchynet":
                layer, pe = _entropy_exit(exit_test, i, ent_taus)
            elif method in ("crc_exit", "coda_risk"):
                layer, pe = _conformal_exit(exit_test, i, crc_t)
            else:  # conformal_exit / coda_sc
                layer, pe = _conformal_exit(exit_test, i, conf_taus)

            if layer is not None:
                preds[t] = int(pe)
                lat[t], en[t] = exit_cost(layer)
                up[t], kind[t] = 0.0, "exit"
                accepted[t] = True
                sel_err[t] = pe != y[i]
            else:
                if method in ("coda_sc", "coda_risk"):
                    a_used = controller_choice(actions + [local_full])   # coupled: offload vs local fallback
                else:
                    feas = [a for a in actions if est_latency(a) <= deadline]
                    a_used = min(feas, key=action_energy) if feas else min(actions, key=est_latency)
                preds[t] = pred_for_action(a_used, i)
                if a_used.payload_kb <= 0:  # local fallback
                    lat[t], up[t], en[t], kind[t] = a_used.local_ms, 0.0, action_energy(a_used), "local"
                else:
                    lat[t], up[t], en[t], kind[t] = action_latency(a_used, trace, t), a_used.payload_kb, action_energy(a_used), "offload"
                record_action(a_used)
        elif method == "aodpart":
            # AODPart-style reference: online delay-constrained accuracy maximisation over
            # {exits, splits}. AODPart is accuracy-driven (not conformal): it exits on a
            # confidence criterion and otherwise offloads through the fastest feasible
            # partition, with a local fallback when nothing is feasible.
            layer, pe = _confidence_exit(exit_test, i, cnf_taus)
            # exit arms feasible if their (deterministic) local latency <= deadline
            exit_feasible = layer is not None and exits[layer]["local_ms"] <= deadline
            if exit_feasible:
                preds[t] = int(pe); lat[t], en[t] = exit_cost(layer); up[t], kind[t] = 0.0, "exit"
                accepted[t] = True; sel_err[t] = pe != y[i]
            else:
                feas = [a for a in actions if est_latency(a) <= deadline]
                a_used = min(feas, key=lambda a: est_latency(a)) if feas else min(actions + [local_full], key=est_latency)
                preds[t] = pred_for_action(a_used, i)
                if a_used.payload_kb <= 0:
                    lat[t], up[t], en[t], kind[t] = a_used.local_ms, 0.0, action_energy(a_used), "local"
                else:
                    lat[t], up[t], en[t], kind[t] = action_latency(a_used, trace, t), a_used.payload_kb, action_energy(a_used), "offload"
                record_action(a_used)
        else:
            raise ValueError(method)

        # dual / EMA updates after observing the realised outcome
        if method in ("coda_sc", "coda_risk", "aodpart", "coda_controller", "coda_split_controller"):
            miss = 1.0 if lat[t] > deadline else 0.0
            dual = max(0.0, dual + ETA_DUAL * (miss - miss_budget))
        if up[t] > 0:
            bw = 0.9 * bw + 0.1 * trace.bandwidth_mbps[t]
            rtt = 0.9 * rtt + 0.1 * trace.rtt_ms[t]
            q = 0.9 * q + 0.1 * trace.queue_ms[t]

    yv = y[order]
    out = dict(
        method=method,
        profile=trace.profile,
        accuracy=float(np.mean(preds == yv)),
        mean_latency_ms=float(np.mean(lat)),
        p95_latency_ms=float(np.percentile(lat, 95)),
        deadline_miss_rate=float(np.mean(lat > deadline)),
        mean_upload_kb=float(np.mean(up)),
        mean_energy_j=float(np.mean(en)),
        exit_rate=float(np.mean(accepted)),
        offload_rate=float(np.mean(kind == "offload")),
        local_rate=float(np.mean(kind == "local")),
        selective_error=float(np.sum(sel_err) / max(1, np.sum(accepted))),
    )
    for name, count in action_counts.items():
        out[f"action_{name}_rate"] = float(count / n)
    return out


def choose_static_action(actions, trace, horizon=300):
    horizon = min(horizon, len(trace.bandwidth_mbps))
    return min(actions, key=lambda a: float(np.mean([action_latency(a, trace, t) for t in range(horizon)])))


def calibration_diagnostics(exit_test, y_test, taus):
    rows = []
    for layer, (probs, tau) in enumerate(zip(exit_test, taus), start=1):
        in_set = (1.0 - probs) <= tau
        sizes = in_set.sum(axis=1)
        singleton = sizes == 1
        pred = probs.argmax(axis=1)
        cover = (1.0 - probs[np.arange(len(y_test)), y_test]) <= tau
        rows.append(dict(
            exit_layer=layer,
            coverage=float(np.mean(cover)),
            avg_set_size=float(np.mean(sizes)),
            singleton_rate=float(np.mean(singleton)),
            singleton_error=float(np.mean(pred[singleton] != y_test[singleton])) if singleton.any() else 0.0,
        ))
    return rows


ALL_METHODS = [
    "local_only", "cloud_only", "static_split", "oracle_split",
    "confidence_exit", "branchynet", "deadline_greedy", "aodpart",
    "conformal_exit", "crc_exit", "coda_sc", "coda_risk",
]


def run_dataset_seed(cache_dir, profile, dataset, seed, alpha=0.10, repeats=6,
                     target_sel_err=0.05, methods=None, network_scale=1.0,
                     bottleneck_channels: int | None = None):
    methods = methods or ALL_METHODS
    bundle = load_bundle(cache_dir, dataset, seed, bottleneck_channels=bottleneck_channels)
    actions, local_full, exits, deadline = actions_from_profile(profile)
    conf_taus = conformal_taus(bundle["exit_cal"], bundle["y_cal"], alpha)
    cnf_taus = confidence_taus_matched(bundle["exit_cal"], bundle["y_cal"], target_sel_err)
    ent_taus = entropy_taus_matched(bundle["exit_cal"], bundle["y_cal"], target_sel_err)
    crc_t = crc_taus(bundle["exit_cal"], bundle["y_cal"], target_sel_err)

    n_test = len(bundle["y_test"])
    order = _repeat_stream(n_test, repeats, seed + 10)
    rows = []
    for prof in PROFILES:
        tseed = seed + DATASET_OFFSETS[dataset] + PROFILE_OFFSETS[prof]
        trace = generate_trace(prof, len(order), tseed)
        static_action = choose_static_action(actions, trace)
        for method in methods:
            r = evaluate_policy(method, bundle, order, trace, actions, local_full, exits, deadline,
                                conf_taus, cnf_taus, ent_taus, crc_t, static_action)
            r.update(dataset=dataset, seed=seed, alpha=alpha)
            rows.append(r)
    cal_rows = [dict(dataset=dataset, seed=seed, **c)
                for c in calibration_diagnostics(bundle["exit_test"], bundle["y_test"], conf_taus)]
    return rows, cal_rows
