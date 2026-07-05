"""Legacy utilities and prototype experiments for conformal split computing.

The current manuscript pipeline uses ``coda_cnn.py`` and ``policy_eval.py`` for
the CNN benchmark.  This module is retained for shared action/trace helpers and
for archival comparison with the earlier lightweight sklearn prototype.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.datasets import load_breast_cancer, load_digits, load_wine, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SplitAction:
    name: str
    split: int
    local_ms: float
    cloud_ms: float
    payload_kb: float
    local_j: float
    upload_j_per_kb: float = 0.0012


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    display: str
    hidden_layers: tuple[int, int, int]
    deadline_ms: float
    local_full_ms: float
    local_full_j: float
    actions: tuple[SplitAction, SplitAction, SplitAction, SplitAction]


@dataclass
class Trace:
    bandwidth_mbps: np.ndarray
    rtt_ms: np.ndarray
    queue_ms: np.ndarray
    loss: np.ndarray
    profile: str


@dataclass
class Evaluation:
    dataset: str
    seed: int
    method: str
    profile: str
    accuracy: float
    mean_latency_ms: float
    p95_latency_ms: float
    deadline_miss_rate: float
    mean_upload_kb: float
    mean_energy_j: float
    exit_rate: float
    offload_rate: float
    risk_violation_rate: float


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def dataset_specs() -> dict[str, DatasetSpec]:
    return {
        "digits": DatasetSpec(
            name="digits",
            display="Digits",
            hidden_layers=(128, 64, 32),
            deadline_ms=75.0,
            local_full_ms=24.0,
            local_full_j=0.86,
            actions=(
                SplitAction("cloud_raw", split=0, local_ms=0.0, cloud_ms=8.2, payload_kb=48.0, local_j=0.02),
                SplitAction("split_l1", split=1, local_ms=4.1, cloud_ms=5.4, payload_kb=20.0, local_j=0.18),
                SplitAction("split_l2", split=2, local_ms=8.9, cloud_ms=3.1, payload_kb=8.0, local_j=0.34),
                SplitAction("split_l3", split=3, local_ms=13.4, cloud_ms=1.2, payload_kb=3.0, local_j=0.48),
            ),
        ),
        "synthetic_iot": DatasetSpec(
            name="synthetic_iot",
            display="Synthetic IoT",
            hidden_layers=(96, 48, 24),
            deadline_ms=70.0,
            local_full_ms=19.0,
            local_full_j=0.62,
            actions=(
                SplitAction("cloud_raw", split=0, local_ms=0.0, cloud_ms=7.0, payload_kb=36.0, local_j=0.018),
                SplitAction("split_l1", split=1, local_ms=3.2, cloud_ms=4.8, payload_kb=15.0, local_j=0.13),
                SplitAction("split_l2", split=2, local_ms=6.9, cloud_ms=2.6, payload_kb=5.5, local_j=0.25),
                SplitAction("split_l3", split=3, local_ms=10.4, cloud_ms=1.0, payload_kb=2.2, local_j=0.34),
            ),
        ),
        "breast_cancer": DatasetSpec(
            name="breast_cancer",
            display="Breast Cancer",
            hidden_layers=(64, 32, 16),
            deadline_ms=65.0,
            local_full_ms=13.0,
            local_full_j=0.42,
            actions=(
                SplitAction("cloud_raw", split=0, local_ms=0.0, cloud_ms=4.8, payload_kb=22.0, local_j=0.012),
                SplitAction("split_l1", split=1, local_ms=2.4, cloud_ms=3.0, payload_kb=9.0, local_j=0.08),
                SplitAction("split_l2", split=2, local_ms=5.1, cloud_ms=1.7, payload_kb=3.8, local_j=0.15),
                SplitAction("split_l3", split=3, local_ms=7.8, cloud_ms=0.7, payload_kb=1.4, local_j=0.22),
            ),
        ),
        "wine": DatasetSpec(
            name="wine",
            display="Wine",
            hidden_layers=(48, 24, 12),
            deadline_ms=60.0,
            local_full_ms=11.0,
            local_full_j=0.34,
            actions=(
                SplitAction("cloud_raw", split=0, local_ms=0.0, cloud_ms=4.1, payload_kb=16.0, local_j=0.010),
                SplitAction("split_l1", split=1, local_ms=2.0, cloud_ms=2.5, payload_kb=7.0, local_j=0.06),
                SplitAction("split_l2", split=2, local_ms=4.3, cloud_ms=1.3, payload_kb=2.8, local_j=0.11),
                SplitAction("split_l3", split=3, local_ms=6.4, cloud_ms=0.5, payload_kb=1.1, local_j=0.16),
            ),
        ),
    }


def available_datasets() -> list[str]:
    return list(dataset_specs().keys())


def _scale_train_cal_test(x_train: np.ndarray, x_cal: np.ndarray, x_test: np.ndarray):
    scaler = StandardScaler()
    return scaler.fit_transform(x_train), scaler.transform(x_cal), scaler.transform(x_test)


def load_dataset_splits(dataset: str, seed: int = 7):
    if dataset == "digits":
        data = load_digits()
        x = data.data.astype(np.float64) / 16.0
        y = data.target.astype(int)
    elif dataset == "wine":
        data = load_wine()
        x = data.data.astype(np.float64)
        y = data.target.astype(int)
    elif dataset == "breast_cancer":
        data = load_breast_cancer()
        x = data.data.astype(np.float64)
        y = data.target.astype(int)
    elif dataset == "synthetic_iot":
        x, y = make_classification(
            n_samples=2400,
            n_features=32,
            n_informative=18,
            n_redundant=6,
            n_classes=5,
            class_sep=1.25,
            flip_y=0.02,
            random_state=seed,
        )
        x = x.astype(np.float64)
        y = y.astype(int)
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x, y, test_size=0.4, random_state=seed, stratify=y
    )
    x_cal, x_test, y_cal, y_test = train_test_split(
        x_tmp, y_tmp, test_size=0.5, random_state=seed + 1, stratify=y_tmp
    )
    if dataset != "digits":
        x_train, x_cal, x_test = _scale_train_cal_test(x_train, x_cal, x_test)
    return x_train, x_cal, x_test, y_train, y_cal, y_test


class MultiExitModel:
    def __init__(self, hidden_layers: tuple[int, int, int], seed: int = 7):
        self.seed = seed
        self.backbone = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="lbfgs",
            alpha=1e-4,
            max_iter=650,
            random_state=seed,
        )
        self.probes: list[LogisticRegression] = []
        self.final_head: LogisticRegression | None = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "MultiExitModel":
        self.backbone.fit(x_train, y_train)
        acts = self.hidden_activations(x_train)
        self.probes = []
        for h in acts:
            probe = LogisticRegression(
                C=0.20,
                max_iter=2000,
                solver="lbfgs",
                random_state=self.seed,
            )
            probe.fit(h, y_train)
            self.probes.append(probe)
        self.final_head = LogisticRegression(
            C=6.0,
            max_iter=2000,
            solver="lbfgs",
            random_state=self.seed,
        )
        self.final_head.fit(acts[-1], y_train)
        return self

    def hidden_activations(self, x: np.ndarray) -> list[np.ndarray]:
        h = x
        acts = []
        for weight, bias in zip(self.backbone.coefs_[:-1], self.backbone.intercepts_[:-1]):
            h = _relu(h @ weight + bias)
            acts.append(h)
        return acts

    def final_proba_from_split(self, x_or_h: np.ndarray, split: int) -> np.ndarray:
        if split == 0:
            h = x_or_h
            start = 0
        else:
            h = x_or_h
            start = split
        for layer in range(start, len(self.backbone.coefs_) - 1):
            h = _relu(h @ self.backbone.coefs_[layer] + self.backbone.intercepts_[layer])
        if self.final_head is None:
            logits = h @ self.backbone.coefs_[-1] + self.backbone.intercepts_[-1]
            return softmax(logits, axis=1)
        return self.final_head.predict_proba(h)

    def exit_probas(self, x: np.ndarray) -> list[np.ndarray]:
        hiddens = self.hidden_activations(x)
        return [probe.predict_proba(h) for probe, h in zip(self.probes, hiddens)]


def conformal_thresholds(model: MultiExitModel, x_cal: np.ndarray, y_cal: np.ndarray, alpha: float) -> list[float]:
    thresholds = []
    n = len(y_cal)
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n)
    for probs in model.exit_probas(x_cal):
        scores = 1.0 - probs[np.arange(n), y_cal]
        thresholds.append(float(np.partition(scores, rank - 1)[rank - 1]))
    return thresholds


def confidence_thresholds(
    model: MultiExitModel,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    target_accuracy: float,
) -> list[float]:
    thresholds = []
    for probs in model.exit_probas(x_cal):
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        best_tau = 0.995
        best_cov = -1
        for tau in np.linspace(0.50, 0.995, 100):
            mask = conf >= tau
            if not np.any(mask):
                continue
            acc = accuracy_score(y_cal[mask], pred[mask])
            cov = int(mask.sum())
            if acc >= target_accuracy and cov > best_cov:
                best_tau = float(tau)
                best_cov = cov
        thresholds.append(best_tau)
    return thresholds


def local_full_action(spec: DatasetSpec) -> SplitAction:
    return SplitAction("local_full", split=4, local_ms=spec.local_full_ms, cloud_ms=0.0, payload_kb=0.0, local_j=spec.local_full_j)


def generate_trace(profile: str, n: int, seed: int) -> Trace:
    rng = np.random.default_rng(seed)
    profiles = {
        "wifi": (38.0, 18.0, 0.30, 0.20, 0.002),
        "lte": (12.0, 47.0, 0.45, 0.35, 0.006),
        "congested": (4.5, 88.0, 0.65, 0.55, 0.018),
        "mixed": (16.0, 55.0, 0.80, 0.60, 0.010),
    }
    if profile not in profiles:
        raise ValueError(f"unknown trace profile: {profile}")

    bw_base, rtt_base, bw_sigma, rtt_sigma, loss_base = profiles[profile]
    bandwidth = np.empty(n)
    rtt = np.empty(n)
    queue = np.empty(n)
    loss = np.empty(n)
    state = 0
    for t in range(n):
        if profile == "mixed" and rng.random() < 0.08:
            state = int(rng.choice([0, 1, 2], p=[0.55, 0.30, 0.15]))
        state_scale = [1.0, 0.45, 0.18][state] if profile == "mixed" else 1.0
        bw = bw_base * state_scale * rng.lognormal(mean=0.0, sigma=bw_sigma)
        rt = rtt_base / max(state_scale, 0.2) * rng.lognormal(mean=0.0, sigma=rtt_sigma)
        if rng.random() < (0.03 if profile != "wifi" else 0.01):
            rt *= rng.uniform(2.5, 5.5)
        bandwidth[t] = max(0.35, bw)
        rtt[t] = max(4.0, rt)
        queue[t] = rng.gamma(shape=1.4, scale=(2.5 + 13.0 * (1.0 / max(state_scale, 0.25))))
        loss[t] = min(0.25, loss_base * rng.lognormal(mean=0.0, sigma=0.7))
    return Trace(bandwidth, rtt, queue, loss, profile)


def action_latency(action: SplitAction, trace: Trace, t: int) -> float:
    if action.payload_kb <= 0:
        return action.local_ms
    retransmit = 1.0 / max(1e-4, 1.0 - trace.loss[t])
    tx_ms = action.payload_kb * 8.0 / max(trace.bandwidth_mbps[t], 1e-6)
    return action.local_ms + action.cloud_ms + trace.rtt_ms[t] + trace.queue_ms[t] + retransmit * tx_ms


def action_energy(action: SplitAction) -> float:
    return action.local_j + action.upload_j_per_kb * action.payload_kb


def repeat_stream(x: np.ndarray, y: np.ndarray, repeats: int, seed: int):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for _ in range(repeats):
        idx = rng.permutation(len(y))
        xs.append(x[idx])
        ys.append(y[idx])
    return np.vstack(xs), np.concatenate(ys)


def _first_conformal_exit(exit_probs: list[np.ndarray], i: int, thresholds: list[float]) -> tuple[int | None, int | None]:
    for layer, (probs, tau) in enumerate(zip(exit_probs, thresholds), start=1):
        pred_set = np.flatnonzero((1.0 - probs[i]) <= tau)
        if len(pred_set) == 1:
            return layer, int(pred_set[0])
    return None, None


def _first_confidence_exit(exit_probs: list[np.ndarray], i: int, thresholds: list[float]) -> tuple[int | None, int | None]:
    for layer, (probs, tau) in enumerate(zip(exit_probs, thresholds), start=1):
        if float(np.max(probs[i])) >= tau:
            return layer, int(np.argmax(probs[i]))
    return None, None


def _predict_offload(model: MultiExitModel, hiddens: list[np.ndarray], x: np.ndarray, i: int, action: SplitAction) -> int:
    if action.split == 0:
        probs = model.final_proba_from_split(x[i : i + 1], 0)
    else:
        probs = model.final_proba_from_split(hiddens[action.split - 1][i : i + 1], action.split)
    return int(np.argmax(probs[0]))


def _exit_costs(spec: DatasetSpec, exit_layer: int) -> tuple[float, float]:
    action = spec.actions[exit_layer]
    return action.local_ms, action.local_j


def evaluate_policy(
    dataset: str,
    seed: int,
    method: str,
    model: MultiExitModel,
    x: np.ndarray,
    y: np.ndarray,
    trace: Trace,
    spec: DatasetSpec,
    conformal_taus: list[float],
    confidence_taus: list[float],
    static_action: SplitAction | None = None,
    miss_budget: float = 0.08,
) -> Evaluation:
    n = len(y)
    actions = list(spec.actions)
    hiddens = model.hidden_activations(x)
    exit_probs = [probe.predict_proba(h) for probe, h in zip(model.probes, hiddens)]
    local_full = local_full_action(spec)
    preds = np.empty(n, dtype=int)
    latencies = np.empty(n)
    uploads = np.empty(n)
    energies = np.empty(n)
    exits = np.zeros(n, dtype=bool)
    offloads = np.zeros(n, dtype=bool)
    singleton_errors = np.zeros(n, dtype=bool)

    bw_hat = float(np.median(trace.bandwidth_mbps[: min(20, n)]))
    rtt_hat = float(np.median(trace.rtt_ms[: min(20, n)]))
    queue_hat = float(np.median(trace.queue_ms[: min(20, n)]))
    dual = 0.0
    eta = 0.025

    def estimated_latency(action: SplitAction) -> float:
        if action.payload_kb <= 0:
            return action.local_ms
        tx_ms = action.payload_kb * 8.0 / max(bw_hat, 1e-6)
        return action.local_ms + action.cloud_ms + rtt_hat + queue_hat + tx_ms

    def offload_with_action(action: SplitAction, i: int) -> tuple[int, float, float, float]:
        pred = _predict_offload(model, hiddens, x, i, action)
        latency = action_latency(action, trace, i)
        return pred, latency, action.payload_kb, action_energy(action)

    for i in range(n):
        if method == "local_only":
            pred = _predict_offload(model, hiddens, x, i, SplitAction("local_proxy", 0, 0, 0, 0, 0))
            latency = local_full.local_ms
            upload = 0.0
            energy = action_energy(local_full)
        elif method == "cloud_only":
            pred, latency, upload, energy = offload_with_action(actions[0], i)
            offloads[i] = True
        elif method == "static_split":
            pred, latency, upload, energy = offload_with_action(static_action or actions[0], i)
            offloads[i] = True
        elif method == "oracle_split":
            action = min(actions, key=lambda a: action_latency(a, trace, i))
            pred, latency, upload, energy = offload_with_action(action, i)
            offloads[i] = True
        elif method in {"confidence_exit", "conformal_exit", "coda_sc"}:
            if method == "confidence_exit":
                exit_layer, pred_exit = _first_confidence_exit(exit_probs, i, confidence_taus)
            else:
                exit_layer, pred_exit = _first_conformal_exit(exit_probs, i, conformal_taus)

            if exit_layer is not None:
                pred = int(pred_exit)
                latency, energy = _exit_costs(spec, exit_layer)
                upload = 0.0
                exits[i] = True
                singleton_errors[i] = pred != y[i]
            else:
                if method == "coda_sc":
                    def objective(a: SplitAction) -> float:
                        est = estimated_latency(a)
                        violation = max(0.0, est - spec.deadline_ms)
                        return est + dual * violation + 8.0 * action_energy(a)

                    action = min(actions, key=objective)
                else:
                    feasible = [a for a in actions if estimated_latency(a) <= spec.deadline_ms]
                    action = min(feasible, key=action_energy) if feasible else min(actions, key=estimated_latency)
                pred, latency, upload, energy = offload_with_action(action, i)
                offloads[i] = True
                if method == "coda_sc":
                    violation = 1.0 if latency > spec.deadline_ms else 0.0
                    dual = max(0.0, dual + eta * (violation - miss_budget))
        elif method == "deadline_greedy":
            feasible = [a for a in actions if estimated_latency(a) <= spec.deadline_ms]
            action = min(feasible, key=action_energy) if feasible else min(actions, key=estimated_latency)
            pred, latency, upload, energy = offload_with_action(action, i)
            offloads[i] = True
        else:
            raise ValueError(f"unknown policy: {method}")

        preds[i] = pred
        latencies[i] = latency
        uploads[i] = upload
        energies[i] = energy

        if upload > 0:
            bw_hat = 0.92 * bw_hat + 0.08 * trace.bandwidth_mbps[i]
            rtt_hat = 0.92 * rtt_hat + 0.08 * trace.rtt_ms[i]
            queue_hat = 0.92 * queue_hat + 0.08 * trace.queue_ms[i]

    return Evaluation(
        dataset=dataset,
        seed=seed,
        method=method,
        profile=trace.profile,
        accuracy=float(np.mean(preds == y)),
        mean_latency_ms=float(np.mean(latencies)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        deadline_miss_rate=float(np.mean(latencies > spec.deadline_ms)),
        mean_upload_kb=float(np.mean(uploads)),
        mean_energy_j=float(np.mean(energies)),
        exit_rate=float(np.mean(exits)),
        offload_rate=float(np.mean(offloads)),
        risk_violation_rate=float(np.sum(singleton_errors) / max(1, np.sum(exits))),
    )


def choose_static_action(actions: list[SplitAction], trace: Trace, horizon: int = 300) -> SplitAction:
    horizon = min(horizon, len(trace.bandwidth_mbps))
    return min(actions, key=lambda a: np.mean([action_latency(a, trace, t) for t in range(horizon)]))


def calibration_diagnostics(
    dataset: str,
    seed: int,
    model: MultiExitModel,
    x_test: np.ndarray,
    y_test: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, float | int | str]]:
    rows = []
    for layer, (probs, tau) in enumerate(zip(model.exit_probas(x_test), thresholds), start=1):
        set_sizes = np.sum((1.0 - probs) <= tau, axis=1)
        true_in_set = (1.0 - probs[np.arange(len(y_test)), y_test]) <= tau
        singleton = set_sizes == 1
        pred = probs.argmax(axis=1)
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "exit_layer": layer,
                "coverage": float(np.mean(true_in_set)),
                "avg_set_size": float(np.mean(set_sizes)),
                "singleton_rate": float(np.mean(singleton)),
                "singleton_error": float(np.mean(pred[singleton] != y_test[singleton])) if np.any(singleton) else 0.0,
            }
        )
    return rows


def run_all(
    seeds: list[int],
    output: Path,
    datasets: list[str] | None = None,
    repeats: int = 6,
    alpha: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = dataset_specs()
    datasets = datasets or available_datasets()
    rows = []
    calibration_rows = []
    profile_offsets = {"wifi": 101, "lte": 211, "congested": 307, "mixed": 419}
    dataset_offsets = {"digits": 1000, "synthetic_iot": 2000, "breast_cancer": 3000, "wine": 4000}
    methods = [
        "local_only",
        "cloud_only",
        "static_split",
        "oracle_split",
        "confidence_exit",
        "deadline_greedy",
        "conformal_exit",
        "coda_sc",
    ]

    for dataset in datasets:
        spec = specs[dataset]
        for seed in seeds:
            x_train, x_cal, x_test, y_train, y_cal, y_test = load_dataset_splits(dataset, seed)
            model = MultiExitModel(spec.hidden_layers, seed).fit(x_train, y_train)
            conformal_taus = conformal_thresholds(model, x_cal, y_cal, alpha)
            confidence_taus = confidence_thresholds(model, x_cal, y_cal, 0.99)
            calibration_rows.extend(calibration_diagnostics(dataset, seed, model, x_test, y_test, conformal_taus))
            x_stream, y_stream = repeat_stream(x_test, y_test, repeats=repeats, seed=seed + 10)
            for profile in ["wifi", "lte", "congested", "mixed"]:
                trace_seed = seed + dataset_offsets[dataset] + profile_offsets[profile]
                trace = generate_trace(profile, len(y_stream), trace_seed)
                static_action = choose_static_action(list(spec.actions), trace)
                for method in methods:
                    rows.append(
                        evaluate_policy(
                            dataset,
                            seed,
                            method,
                            model,
                            x_stream,
                            y_stream,
                            trace,
                            spec,
                            conformal_taus,
                            confidence_taus,
                            static_action=static_action,
                        ).__dict__
                    )

    df = pd.DataFrame(rows)
    cal_df = pd.DataFrame(calibration_rows)
    output.mkdir(parents=True, exist_ok=True)
    df.to_csv(output / "summary.csv", index=False)
    cal_df.to_csv(output / "calibration.csv", index=False)
    write_latex_tables(df, cal_df, output)
    make_plots(df, output)
    make_graphical_abstract(output)
    return df, cal_df


def _method_order() -> list[str]:
    return [
        "local_only",
        "cloud_only",
        "static_split",
        "oracle_split",
        "confidence_exit",
        "deadline_greedy",
        "conformal_exit",
        "coda_sc",
    ]


def _method_names() -> dict[str, str]:
    return {
        "local_only": "Local only",
        "cloud_only": "Cloud only",
        "static_split": "Static split",
        "oracle_split": "Oracle split",
        "confidence_exit": "Confidence exit",
        "deadline_greedy": "Deadline greedy",
        "conformal_exit": "Conformal exit",
        "coda_sc": "CODA-SC",
    }


def write_method_means(df: pd.DataFrame, path: Path) -> None:
    metric_cols = [
        "accuracy",
        "mean_latency_ms",
        "p95_latency_ms",
        "deadline_miss_rate",
        "mean_upload_kb",
        "mean_energy_j",
        "exit_rate",
        "risk_violation_rate",
    ]
    means = df.groupby("method", as_index=False)[metric_cols].mean()
    means.to_csv(path, index=False)


def write_latex_tables(df: pd.DataFrame, cal_df: pd.DataFrame, output: Path) -> None:
    write_method_means(df, output / "method_means.csv")
    write_summary_table(df, output / "summary_table.tex")
    write_dataset_table(df, output / "dataset_table.tex")
    write_ablation_table(df, output / "ablation_table.tex")
    write_calibration_table(cal_df, output / "calibration_table.tex")


def write_summary_table(df: pd.DataFrame, path: Path) -> None:
    names = _method_names()
    rows = []
    for method in _method_order():
        sub = df[df["method"] == method]
        rows.append(
            (
                names[method],
                100 * sub["accuracy"].mean(),
                sub["mean_latency_ms"].mean(),
                sub["p95_latency_ms"].mean(),
                100 * sub["deadline_miss_rate"].mean(),
                sub["mean_upload_kb"].mean(),
                sub["mean_energy_j"].mean(),
                100 * sub["exit_rate"].mean(),
                100 * sub["risk_violation_rate"].mean(),
            )
        )
    lines = [
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Method & Acc. (\\%) & Mean ms & P95 ms & Miss (\\%) & Upload KB & Energy J & Exit (\\%) & Sel. err. (\\%) \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row[0]} & {row[1]:.2f} & {row[2]:.1f} & {row[3]:.1f} & {row[4]:.1f} & {row[5]:.1f} & {row[6]:.2f} & {row[7]:.1f} & {row[8]:.1f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_dataset_table(df: pd.DataFrame, path: Path) -> None:
    specs = dataset_specs()
    keep = ["cloud_only", "static_split", "confidence_exit", "coda_sc"]
    names = _method_names()
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Dataset & Method & Acc. (\\%) & Mean ms & Miss (\\%) & Upload KB \\\\",
        "\\midrule",
    ]
    for dataset in available_datasets():
        for method in keep:
            sub = df[(df["dataset"] == dataset) & (df["method"] == method)]
            lines.append(
                f"{specs[dataset].display} & {names[method]} & {100*sub['accuracy'].mean():.2f} & {sub['mean_latency_ms'].mean():.1f} & {100*sub['deadline_miss_rate'].mean():.1f} & {sub['mean_upload_kb'].mean():.1f} \\\\"
            )
        lines.append("\\addlinespace")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_ablation_table(df: pd.DataFrame, path: Path) -> None:
    keep = ["deadline_greedy", "conformal_exit", "coda_sc"]
    names = _method_names()
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Variant & Mean ms & P95 ms & Miss (\\%) & Upload KB & Exit (\\%) \\\\",
        "\\midrule",
    ]
    for method in keep:
        sub = df[df["method"] == method]
        lines.append(
            f"{names[method]} & {sub['mean_latency_ms'].mean():.1f} & {sub['p95_latency_ms'].mean():.1f} & {100*sub['deadline_miss_rate'].mean():.1f} & {sub['mean_upload_kb'].mean():.1f} & {100*sub['exit_rate'].mean():.1f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_calibration_table(cal_df: pd.DataFrame, path: Path) -> None:
    grouped = cal_df.groupby("exit_layer", as_index=False).mean(numeric_only=True)
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Exit & Coverage (\\%) & Avg. set size & Singleton (\\%) & Singleton err. (\\%) \\\\",
        "\\midrule",
    ]
    for _, row in grouped.iterrows():
        lines.append(
            f"Layer {int(row['exit_layer'])} & {100*row['coverage']:.1f} & {row['avg_set_size']:.2f} & {100*row['singleton_rate']:.1f} & {100*row['singleton_error']:.1f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def make_plots(df: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    names = _method_names()
    colors = {
        "local_only": "#7a7a7a",
        "cloud_only": "#4f77b3",
        "static_split": "#5ca36f",
        "oracle_split": "#8b6ab3",
        "confidence_exit": "#d28f3d",
        "deadline_greedy": "#b85c5c",
        "conformal_exit": "#879c48",
        "coda_sc": "#3b8c91",
    }
    means = df.groupby("method", as_index=False).mean(numeric_only=True)
    means = means.set_index("method").loc[_method_order()].reset_index()

    plt.figure(figsize=(7.2, 4.2))
    for _, row in means.iterrows():
        plt.scatter(
            row["mean_latency_ms"],
            100 * row["accuracy"],
            s=105,
            label=names[row["method"]],
            color=colors[row["method"]],
            edgecolor="black",
            linewidth=0.5,
        )
    plt.xlabel("Mean latency (ms)")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False, ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output / "latency_accuracy.pdf")
    plt.savefig(output / "latency_accuracy.png", dpi=180)
    plt.close()

    prof = df[df["method"].isin(["cloud_only", "static_split", "confidence_exit", "conformal_exit", "coda_sc"])]
    pivot = prof.groupby(["profile", "method"])["p95_latency_ms"].mean().unstack()
    pivot = pivot.loc[["wifi", "lte", "congested", "mixed"], ["cloud_only", "static_split", "confidence_exit", "conformal_exit", "coda_sc"]]
    ax = pivot.rename(columns=names).plot(kind="bar", figsize=(7.6, 4.2), color=[colors[c] for c in pivot.columns])
    ax.set_ylabel("P95 latency (ms)")
    ax.set_xlabel("")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output / "p95_by_profile.pdf")
    plt.savefig(output / "p95_by_profile.png", dpi=180)
    plt.close()


def make_graphical_abstract(output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    fig, ax = plt.subplots(figsize=(13.28, 5.31))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    colors = ["#edf4f7", "#eef6ef", "#fff5e6", "#edf4f7"]
    labels = [
        ("Input stream", "Device receives data under\\nchanging Internet state"),
        ("Conformal exits", "Singleton prediction set:\\nanswer locally"),
        ("Online split control", "Uncertain samples choose\\ndeadline-aware split"),
        ("Outcome", "Lower upload and fewer\\ndeadline misses"),
    ]
    xs = [0.04, 0.29, 0.54, 0.79]
    for x, color, (title, body) in zip(xs, colors, labels):
        ax.add_patch(Rectangle((x, 0.24), 0.18, 0.52, facecolor=color, edgecolor="#2f3a3d", linewidth=1.5))
        ax.text(x + 0.09, 0.61, title, ha="center", va="center", fontsize=16, fontweight="bold", color="#253033")
        ax.text(x + 0.09, 0.43, body, ha="center", va="center", fontsize=12, color="#253033")
    for x0, x1 in zip(xs[:-1], xs[1:]):
        ax.add_patch(FancyArrowPatch((x0 + 0.18, 0.50), (x1, 0.50), arrowstyle="-|>", mutation_scale=20, linewidth=1.5, color="#506066"))
    ax.text(0.5, 0.90, "CODA-SC: reliable split computing for intelligent systems", ha="center", fontsize=20, fontweight="bold", color="#1f2c2f")
    ax.text(0.5, 0.12, "Calibration controls early-exit reliability; online control limits Internet-tail latency.", ha="center", fontsize=13, color="#3b4b50")
    plt.tight_layout()
    plt.savefig(output / "graphical_abstract.png", dpi=100)
    plt.savefig(output / "graphical_abstract.pdf")
    plt.close(fig)
