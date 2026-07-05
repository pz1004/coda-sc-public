"""Aggregation, statistics, LaTeX tables, and figures for the CODA-SC revision.

Consumes the long-form per-run DataFrame produced by ``policy_eval`` and emits
the tables/figures cited by the manuscript, including 95% confidence intervals
and paired Wilcoxon signed-rank tests for the key CODA-SC comparisons.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METHOD_NAMES = {
    "local_only": "Local only",
    "cloud_only": "Cloud only",
    "static_split": "Static split",
    "oracle_split": "Oracle split",
    "confidence_exit": "Confidence exit",
    "branchynet": "BranchyNet exit",
    "deadline_greedy": "Deadline greedy",
    "aodpart": "AODPart-style",
    "conformal_exit": "Conformal exit",
    "crc_exit": "CRC exit",
    "coda_sc": "CODA-SC-Cov",
    "coda_risk": "CODA-SC-Risk",
    "coda_controller": "CODA controller (no exits)",
    "coda_split_controller": "CODA split controller (no exits)",
}
MAIN_ORDER = ["local_only", "cloud_only", "static_split", "oracle_split", "confidence_exit",
              "branchynet", "deadline_greedy", "aodpart", "conformal_exit", "crc_exit",
              "coda_sc", "coda_risk"]
COLORS = {
    "local_only": "#7a7a7a", "cloud_only": "#4f77b3", "static_split": "#5ca36f",
    "oracle_split": "#8b6ab3", "confidence_exit": "#d28f3d", "branchynet": "#c26f9c",
    "deadline_greedy": "#b85c5c", "aodpart": "#6f8f3d", "conformal_exit": "#879c48",
    "crc_exit": "#4aa3a0", "coda_sc": "#2f6f74", "coda_risk": "#7d4e8f",
}


def _ci95(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if len(x) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def paired_wilcoxon(df: pd.DataFrame, ref: str, other: str, metric: str) -> float:
    """Paired signed-rank p-value over matched (dataset, seed, profile) cells."""
    keys = ["dataset", "seed", "profile"]
    a = df[df.method == ref].set_index(keys)[metric]
    b = df[df.method == other].set_index(keys)[metric]
    a, b = a.align(b, join="inner")
    d = (a - b).to_numpy()
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(a.to_numpy(), b.to_numpy(), zero_method="zsplit").pvalue)
    except ValueError:
        return 1.0


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _fmt(v, p=1):
    return f"{v:.{p}f}"


def write_summary_table(df: pd.DataFrame, path: Path) -> None:
    g = df.groupby("method")
    lines = [
        "\\begin{tabular}{lrrrrrrrrrr}", "\\toprule",
        "Method & Acc. (\\%) & Mean ms & P95 ms & Miss (\\%) & Upload KB & Energy mJ & Exit (\\%) & Local (\\%) & Offload (\\%) & Sel. err. (\\%) \\\\",
        "\\midrule",
    ]
    for m in MAIN_ORDER:
        s = g.get_group(m)
        lines.append(
            f"{METHOD_NAMES[m]} & {100*s.accuracy.mean():.2f} & {s.mean_latency_ms.mean():.1f} & "
            f"{s.p95_latency_ms.mean():.1f} & {100*s.deadline_miss_rate.mean():.2f} & "
            f"{s.mean_upload_kb.mean():.2f} & {1000*s.mean_energy_j.mean():.1f} & "
            f"{100*s.exit_rate.mean():.1f} & {100*s.local_rate.mean():.1f} & "
            f"{100*s.offload_rate.mean():.1f} & {100*s.selective_error.mean():.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_stats_table(df: pd.DataFrame, path: Path) -> None:
    """95% CIs for headline metrics + Wilcoxon p vs CODA-SC-Cov."""
    keys = ["dataset", "seed", "profile"]
    methods = ["confidence_exit", "aodpart", "conformal_exit", "coda_sc", "coda_risk"]
    g = df.groupby("method")
    lines = [
        "\\begin{tabular}{lrrrrr}", "\\toprule",
        "Method & Mean ms (95\\% CI) & Miss \\% (95\\% CI) & Upload KB (95\\% CI) & Sel. err. \\% (95\\% CI) & $p$ vs Cov \\\\",
        "\\midrule",
    ]
    for m in methods:
        s = g.get_group(m)
        # cell-level samples for CI (per dataset,seed,profile)
        lat = s.groupby(keys).mean_latency_ms.mean()
        miss = 100 * s.groupby(keys).deadline_miss_rate.mean()
        upl = s.groupby(keys).mean_upload_kb.mean()
        sel = 100 * s.groupby(keys).selective_error.mean()
        if m == "coda_sc":
            ptxt = "--"
        else:
            p_lat = paired_wilcoxon(df, "coda_sc", m, "mean_latency_ms")
            p_miss = paired_wilcoxon(df, "coda_sc", m, "deadline_miss_rate")
            ptxt = f"lat {p_lat:.1e}, miss {p_miss:.1e}"
        lines.append(
            f"{METHOD_NAMES[m]} & {lat.mean():.1f}$\\pm${_ci95(lat):.1f} & "
            f"{miss.mean():.2f}$\\pm${_ci95(miss):.2f} & {upl.mean():.2f}$\\pm${_ci95(upl):.2f} & "
            f"{sel.mean():.2f}$\\pm${_ci95(sel):.2f} & {ptxt} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_dataset_table(df: pd.DataFrame, path: Path, displays: dict) -> None:
    keep = ["cloud_only", "confidence_exit", "aodpart", "conformal_exit", "coda_sc", "coda_risk"]
    lines = [
        "\\begin{tabular}{llrrrrr}", "\\toprule",
        "Dataset & Method & Acc. (\\%) & Mean ms & Miss (\\%) & Upload KB & Exit (\\%) \\\\", "\\midrule",
    ]
    for ds in ["mnist", "fashion_mnist", "cifar10", "svhn"]:
        for m in keep:
            s = df[(df.dataset == ds) & (df.method == m)]
            lines.append(
                f"{displays[ds]} & {METHOD_NAMES[m]} & {100*s.accuracy.mean():.2f} & {s.mean_latency_ms.mean():.1f} & "
                f"{100*s.deadline_miss_rate.mean():.1f} & {s.mean_upload_kb.mean():.2f} & {100*s.exit_rate.mean():.1f} \\\\"
            )
        lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_ablation_table(df: pd.DataFrame, path: Path) -> None:
    keep = ["deadline_greedy", "confidence_exit", "conformal_exit", "coda_sc", "coda_risk"]
    lines = [
        "\\begin{tabular}{lrrrrrrr}", "\\toprule",
        "Variant & Mean ms & P95 ms & Miss (\\%) & Upload KB & Energy mJ & Exit (\\%) & Sel. err. (\\%) \\\\", "\\midrule",
    ]
    for m in keep:
        s = df[df.method == m]
        lines.append(
            f"{METHOD_NAMES[m]} & {s.mean_latency_ms.mean():.1f} & {s.p95_latency_ms.mean():.1f} & "
            f"{100*s.deadline_miss_rate.mean():.2f} & {s.mean_upload_kb.mean():.2f} & "
            f"{1000*s.mean_energy_j.mean():.1f} & {100*s.exit_rate.mean():.1f} & "
            f"{100*s.selective_error.mean():.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_calibration_table(cal_df: pd.DataFrame, path: Path) -> None:
    g = cal_df.groupby("exit_layer", as_index=False).mean(numeric_only=True)
    lines = [
        "\\begin{tabular}{lrrrr}", "\\toprule",
        "Exit & Coverage (\\%) & Avg. set size & Singleton (\\%) & Singleton err. (\\%) \\\\", "\\midrule",
    ]
    for _, r in g.iterrows():
        lines.append(
            f"Layer {int(r.exit_layer)} & {100*r.coverage:.1f} & {r.avg_set_size:.2f} & "
            f"{100*r.singleton_rate:.1f} & {100*r.singleton_error:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_sensitivity_table(sens: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{tabular}{llrrrr}", "\\toprule",
        "Sweep & Value & Exit (\\%) & Miss (\\%) & Mean ms & Sel. err. (\\%) \\\\", "\\midrule",
    ]
    for sweep in ["alpha", "edge_gflops"]:
        sub = sens[sens.sweep == sweep]
        for _, r in sub.iterrows():
            lab = {"alpha": "$\\alpha$", "edge_gflops": "Edge GFLOP/s"}[sweep]
            lines.append(
                f"{lab} & {r.value:g} & {100*r.exit_rate:.1f} & {100*r.deadline_miss_rate:.1f} & "
                f"{r.mean_latency_ms:.1f} & {100*r.selective_error:.2f} \\\\"
            )
        lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_action_profile_table(df: pd.DataFrame, path: Path) -> None:
    keep = ["aodpart", "coda_sc", "coda_risk"]
    profiles = ["wifi", "lte", "congested", "mixed"]
    lines = [
        "\\begin{tabular}{llrrrrrr}", "\\toprule",
        "Method & Profile & Local (\\%) & Raw (\\%) & Split1 (\\%) & Split2 (\\%) & Split3 (\\%) & Offload (\\%) \\\\",
        "\\midrule",
    ]
    for m in keep:
        for prof in profiles:
            s = df[(df.method == m) & (df.profile == prof)]
            lines.append(
                f"{METHOD_NAMES[m]} & {prof.title()} & {100*s.action_local_full_rate.mean():.2f} & "
                f"{100*s.action_cloud_raw_rate.mean():.2f} & {100*s.action_split_l1_rate.mean():.2f} & "
                f"{100*s.action_split_l2_rate.mean():.2f} & {100*s.action_split_l3_rate.mean():.2f} & "
                f"{100*s.offload_rate.mean():.2f} \\\\"
            )
        lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_compression_table(comp: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{tabular}{lrrrrrrrr}", "\\toprule",
        "Payload scale & Mean ms & Miss (\\%) & Upload KB & Local (\\%) & Raw (\\%) & Split1 (\\%) & Split1$\\mid$ctrl (\\%) & Sel. err. (\\%) \\\\",
        "\\midrule",
    ]
    for _, r in comp.iterrows():
        controlled = (
            r.action_local_full_rate + r.action_cloud_raw_rate + r.action_split_l1_rate
            + r.action_split_l2_rate + r.action_split_l3_rate
        )
        split1_ctrl = 100 * r.action_split_l1_rate / max(controlled, 1e-12)
        lines.append(
            f"{r.payload_scale:g} & {r.mean_latency_ms:.1f} & {100*r.deadline_miss_rate:.2f} & "
            f"{r.mean_upload_kb:.2f} & {100*r.action_local_full_rate:.2f} & "
            f"{100*r.action_cloud_raw_rate:.2f} & {100*r.action_split_l1_rate:.2f} & "
            f"{split1_ctrl:.2f} & {100*r.selective_error:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_split_stress_table(df: pd.DataFrame, path: Path) -> None:
    """High-resolution/bottleneck stress table used to diagnose split selection."""
    keep = ["cloud_only", "deadline_greedy", "oracle_split", "coda_split_controller", "coda_controller"]
    lines = [
        "\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
        "Method & Mean ms & P95 ms & Miss (\\%) & Upload KB & Local (\\%) & Raw (\\%) & Split1 (\\%) & Split2 (\\%) & Split3 (\\%) \\\\",
        "\\midrule",
    ]
    for m in keep:
        s = df[df.method == m]
        lines.append(
            f"{METHOD_NAMES[m]} & {s.mean_latency_ms.mean():.1f} & {s.p95_latency_ms.mean():.1f} & "
            f"{100*s.deadline_miss_rate.mean():.2f} & {s.mean_upload_kb.mean():.2f} & "
            f"{100*s.action_local_full_rate.mean():.1f} & {100*s.action_cloud_raw_rate.mean():.1f} & "
            f"{100*s.action_split_l1_rate.mean():.1f} & {100*s.action_split_l2_rate.mean():.1f} & "
            f"{100*s.action_split_l3_rate.mean():.1f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_bottleneck_table(df: pd.DataFrame, path: Path) -> None:
    """Learned-codec split-selection table used in the main manuscript."""
    keep = ["cloud_only", "deadline_greedy", "oracle_split", "coda_split_controller", "coda_controller"]
    lines = [
        "\\begin{tabular}{lrrrrrrrrrr}", "\\toprule",
        "Method & Acc. (\\%) & Mean ms & P95 ms & Miss (\\%) & Upload KB & Local (\\%) & Raw (\\%) & Split1 (\\%) & Split2 (\\%) & Split3 (\\%) \\\\",
        "\\midrule",
    ]
    for m in keep:
        s = df[df.method == m]
        lines.append(
            f"{METHOD_NAMES[m]} & {100*s.accuracy.mean():.2f} & {s.mean_latency_ms.mean():.1f} & "
            f"{s.p95_latency_ms.mean():.1f} & {100*s.deadline_miss_rate.mean():.2f} & "
            f"{s.mean_upload_kb.mean():.2f} & {100*s.action_local_full_rate.mean():.1f} & "
            f"{100*s.action_cloud_raw_rate.mean():.1f} & {100*s.action_split_l1_rate.mean():.1f} & "
            f"{100*s.action_split_l2_rate.mean():.1f} & {100*s.action_split_l3_rate.mean():.1f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figures(df: pd.DataFrame, cal_df: pd.DataFrame, out: Path, displays: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means = df.groupby("method").mean(numeric_only=True)

    # 1) latency-accuracy trade-off
    plt.figure(figsize=(7.2, 4.4))
    for m in MAIN_ORDER:
        r = means.loc[m]
        plt.scatter(r.mean_latency_ms, 100 * r.accuracy, s=110, color=COLORS[m],
                    edgecolor="black", linewidth=0.5, label=METHOD_NAMES[m].replace("$^\\dagger$", ""))
    plt.xlabel("Mean latency (ms)"); plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.25); plt.legend(frameon=False, ncol=2, fontsize=8)
    plt.tight_layout(); plt.savefig(out / "latency_accuracy.pdf"); plt.savefig(out / "latency_accuracy.png", dpi=180); plt.close()

    # 2) P95 by profile (network-dependent methods)
    net = ["cloud_only", "aodpart", "conformal_exit", "coda_sc", "coda_risk"]
    piv = df[df.method.isin(net)].groupby(["profile", "method"]).p95_latency_ms.mean().unstack()
    piv = piv.loc[["wifi", "lte", "congested", "mixed"], net]
    ax = piv.rename(columns={k: METHOD_NAMES[k].replace("$^\\dagger$", "") for k in net}).plot(
        kind="bar", figsize=(7.4, 4.2), color=[COLORS[c] for c in net])
    ax.set_ylabel("P95 latency (ms)"); ax.set_xlabel("")
    ax.grid(True, axis="y", alpha=0.25); ax.legend(frameon=False, ncol=2, fontsize=8)
    plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(out / "p95_by_profile.pdf"); plt.savefig(out / "p95_by_profile.png", dpi=180); plt.close()

    # 3) deadline-miss distribution (box) across dataset x seed x profile -- the tail story
    plt.figure(figsize=(7.4, 4.2))
    box_methods = ["cloud_only", "aodpart", "conformal_exit", "coda_sc", "coda_risk"]
    data = [100 * df[df.method == m].deadline_miss_rate.to_numpy() for m in box_methods]
    bp = plt.boxplot(data, tick_labels=[METHOD_NAMES[m].replace("$^\\dagger$", "") for m in box_methods],
                     patch_artist=True, showmeans=True)
    for patch, m in zip(bp["boxes"], box_methods):
        patch.set_facecolor(COLORS[m]); patch.set_alpha(0.7)
    plt.ylabel("Deadline miss rate (%)"); plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout(); plt.savefig(out / "miss_box.pdf"); plt.savefig(out / "miss_box.png", dpi=180); plt.close()

    # 4) exit rate vs dataset difficulty
    plt.figure(figsize=(7.0, 4.0))
    exit_methods = ["confidence_exit", "conformal_exit", "coda_sc", "coda_risk"]
    ds_order = ["mnist", "fashion_mnist", "cifar10", "svhn"]
    xw = np.arange(len(ds_order))
    offsets = np.linspace(-0.3, 0.3, len(exit_methods))
    for j, m in enumerate(exit_methods):
        vals = [100 * df[(df.dataset == d) & (df.method == m)].exit_rate.mean() for d in ds_order]
        plt.bar(xw + offsets[j], vals, width=0.18, color=COLORS[m], label=METHOD_NAMES[m].replace("$^\\dagger$", ""))
    plt.xticks(xw, [displays[d] for d in ds_order]); plt.ylabel("Local exit rate (%)")
    plt.grid(True, axis="y", alpha=0.25); plt.legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(out / "exit_by_dataset.pdf"); plt.savefig(out / "exit_by_dataset.png", dpi=180); plt.close()


def make_graphical_abstract(out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    fig, ax = plt.subplots(figsize=(13.28, 5.31)); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    colors = ["#edf4f7", "#eef6ef", "#fff5e6", "#edf4f7"]
    labels = [
        ("Vision stream", "Edge device under\nchanging Internet state"),
        ("Calibrated exits", "Coverage or risk mode:\nanswer locally"),
        ("Coupled control", "Uncertain sample:\noffload or run locally"),
        ("Outcome", "Near-zero deadline\nmisses, low energy"),
    ]
    xs = [0.04, 0.29, 0.54, 0.79]
    for x, color, (title, body) in zip(xs, colors, labels):
        ax.add_patch(Rectangle((x, 0.24), 0.18, 0.52, facecolor=color, edgecolor="#2f3a3d", linewidth=1.5))
        ax.text(x + 0.09, 0.61, title, ha="center", va="center", fontsize=16, fontweight="bold", color="#253033")
        ax.text(x + 0.09, 0.42, body, ha="center", va="center", fontsize=12, color="#253033")
    for x0, x1 in zip(xs[:-1], xs[1:]):
        ax.add_patch(FancyArrowPatch((x0 + 0.18, 0.50), (x1, 0.50), arrowstyle="-|>", mutation_scale=20, linewidth=1.5, color="#506066"))
    ax.text(0.5, 0.90, "CODA-SC: reliable split computing for intelligent systems", ha="center", fontsize=20, fontweight="bold", color="#1f2c2f")
    ax.text(0.5, 0.12, "Measured CNN FLOPs/payloads; modeled devices/traces; calibrated exits + deadline control.", ha="center", fontsize=13, color="#3b4b50")
    plt.tight_layout(); plt.savefig(out / "graphical_abstract.png", dpi=100); plt.savefig(out / "graphical_abstract.pdf"); plt.close(fig)
