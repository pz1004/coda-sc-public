#!/usr/bin/env python3
"""Reproduce the CODA-SC experiments end to end.

Trains the multi-exit CNNs (if not cached), evaluates every policy over the
Internet-like network traces, computes statistics and sensitivity sweeps, and
writes all LaTeX tables and figures used by the manuscript.

    python3 scripts/run_experiments.py --seeds 7,13,29,42,101
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import coda_cnn  # noqa: E402
import policy_eval as PE  # noqa: E402
import report as R  # noqa: E402


def ensure_trained(seeds, data_root, cache_dir, profiles_path, need_weights=False):
    datasets = list(coda_cnn.DATASETS.keys())
    missing = [(d, s) for d in datasets for s in seeds if not (cache_dir / f"{d}_s{s}.npz").exists()]
    missing_weights = []
    if need_weights:
        missing_weights = [(d, s) for d in datasets for s in seeds if not (cache_dir / f"{d}_s{s}.pt").exists()]
    if missing or missing_weights or not profiles_path.exists():
        print(f"training/caching ({len(missing)} probability runs, {len(missing_weights)} weight files missing)...", flush=True)
        coda_cnn.run_training(seeds, data_root, cache_dir, profiles_path)
    return json.loads(profiles_path.read_text())


def sensitivity(profiles, cache_dir, seeds, repeats):
    rows = []
    datasets = list(coda_cnn.DATASETS.keys())
    for a in (0.05, 0.10, 0.15, 0.20):
        acc = []
        for d in datasets:
            for s in seeds:
                r, _ = PE.run_dataset_seed(cache_dir, profiles[d], d, s, alpha=a,
                                           repeats=repeats, methods=["coda_sc"])
                acc.extend(r)
        m = pd.DataFrame(acc).mean(numeric_only=True)
        rows.append(dict(sweep="alpha", value=a, **{k: m[k] for k in
                    ["exit_rate", "deadline_miss_rate", "mean_latency_ms", "selective_error"]}))
    for g in (2.5, 5.0, 10.0, 20.0):
        acc = []
        for d in datasets:
            prof = PE.rescale_profile(profiles[d], g)
            for s in seeds:
                r, _ = PE.run_dataset_seed(cache_dir, prof, d, s, alpha=0.10,
                                           repeats=repeats, methods=["coda_sc"])
                acc.extend(r)
        m = pd.DataFrame(acc).mean(numeric_only=True)
        rows.append(dict(sweep="edge_gflops", value=g, **{k: m[k] for k in
                    ["exit_rate", "deadline_miss_rate", "mean_latency_ms", "selective_error"]}))
    return pd.DataFrame(rows)


def compression_sensitivity(profiles, cache_dir, seeds, repeats, raw_input_scale=128.0):
    rows = []
    datasets = list(coda_cnn.DATASETS.keys())
    for scale in (1.0, 0.5, 0.25, 0.125):
        acc = []
        for d in datasets:
            prof = PE.rescale_payloads(profiles[d], payload_scale=scale, raw_input_scale=raw_input_scale)
            for s in seeds:
                r, _ = PE.run_dataset_seed(cache_dir, prof, d, s, alpha=0.10,
                                           repeats=repeats, methods=["coda_sc"])
                acc.extend(r)
        m = pd.DataFrame(acc).mean(numeric_only=True)
        keep = [
            "mean_latency_ms", "deadline_miss_rate", "mean_upload_kb", "selective_error",
            "action_local_full_rate", "action_cloud_raw_rate", "action_split_l1_rate",
            "action_split_l2_rate", "action_split_l3_rate",
        ]
        rows.append(dict(payload_scale=scale, raw_input_scale=raw_input_scale, **{k: m[k] for k in keep}))
    return pd.DataFrame(rows)


def split_selection_stress(profiles, cache_dir, seeds, repeats,
                           raw_input_scale=128.0, payload_scale=0.125):
    """Stress-test the split controller in a high-resolution/bottleneck regime.

    The trained CNNs and FLOP measurements are unchanged.  Raw-input upload is
    inflated to emulate higher-resolution frames, intermediate activations are
    scaled to emulate a learned bottleneck, and exits are disabled for the
    CODA controller row so that the table isolates the offload/split/fallback
    action space.
    """
    rows = []
    datasets = list(coda_cnn.DATASETS.keys())
    methods = ["cloud_only", "deadline_greedy", "oracle_split",
               "coda_split_controller", "coda_controller"]
    for d in datasets:
        prof = PE.rescale_payloads(profiles[d], payload_scale=payload_scale,
                                   raw_input_scale=raw_input_scale)
        for s in seeds:
            r, _ = PE.run_dataset_seed(cache_dir, prof, d, s, alpha=0.10,
                                       repeats=repeats, methods=methods)
            for row in r:
                row["raw_input_scale"] = raw_input_scale
                row["payload_scale"] = payload_scale
            rows.extend(r)
    return pd.DataFrame(rows)


def learned_bottleneck_experiment(profiles, cache_dir, data_root, seeds, repeats,
                                  bottleneck_channels, bottleneck_epochs, alpha,
                                  raw_input_scale=128.0):
    rows = []
    profiles_out = {}
    datasets = list(coda_cnn.DATASETS.keys())
    methods = ["cloud_only", "deadline_greedy", "oracle_split",
               "coda_split_controller", "coda_controller"]
    for d in datasets:
        profiles_out[d] = {}
        for s in seeds:
            prof = coda_cnn.export_learned_bottlenecks(
                d, s, data_root, cache_dir, profiles[d],
                bottleneck_channels=bottleneck_channels,
                epochs=bottleneck_epochs,
            )
            prof = PE.rescale_payloads(prof, payload_scale=1.0, raw_input_scale=raw_input_scale)
            profiles_out[d][str(s)] = prof
            r, _ = PE.run_dataset_seed(
                cache_dir, prof, d, s, alpha=alpha, repeats=repeats,
                methods=methods, bottleneck_channels=bottleneck_channels,
            )
            for row in r:
                row["bottleneck_channels"] = bottleneck_channels
                row["bottleneck_epochs"] = bottleneck_epochs
                row["raw_input_scale"] = raw_input_scale
            rows.extend(r)
    return pd.DataFrame(rows), profiles_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7,13,29,42,101")
    ap.add_argument("--output", type=Path, default=ROOT / "results")
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--cache", type=Path, default=ROOT / "results" / "cnn_cache")
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--enable-learned-bottlenecks", action="store_true")
    ap.add_argument("--bottleneck-channels", type=int, default=8)
    ap.add_argument("--bottleneck-epochs", type=int, default=8)
    ap.add_argument("--bottleneck-raw-input-scale", type=float, default=128.0)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s]
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    profiles = ensure_trained(
        seeds, args.data_root, args.cache, out / "profiles.json",
        need_weights=args.enable_learned_bottlenecks,
    )
    displays = {d: profiles[d]["display"] for d in profiles}

    all_rows, all_cal = [], []
    for d in coda_cnn.DATASETS:
        for s in seeds:
            rows, cal = PE.run_dataset_seed(args.cache, profiles[d], d, s, alpha=args.alpha, repeats=args.repeats)
            all_rows.extend(rows)
            all_cal.extend(cal)
    df = pd.DataFrame(all_rows)
    cal_df = pd.DataFrame(all_cal)
    df.to_csv(out / "summary.csv", index=False)
    cal_df.to_csv(out / "calibration.csv", index=False)
    df.groupby("method").mean(numeric_only=True).to_csv(out / "method_means.csv")

    R.write_summary_table(df, out / "summary_table.tex")
    R.write_stats_table(df, out / "stats_table.tex")
    R.write_dataset_table(df, out / "dataset_table.tex", displays)
    R.write_ablation_table(df, out / "ablation_table.tex")
    R.write_calibration_table(cal_df, out / "calibration_table.tex")
    R.write_action_profile_table(df, out / "action_profile_table.tex")

    sens = sensitivity(profiles, args.cache, seeds, repeats=max(3, args.repeats // 2))
    sens.to_csv(out / "sensitivity.csv", index=False)
    R.write_sensitivity_table(sens, out / "sensitivity_table.tex")

    comp = compression_sensitivity(profiles, args.cache, seeds, repeats=max(3, args.repeats // 2))
    comp.to_csv(out / "compression_sensitivity.csv", index=False)
    R.write_compression_table(comp, out / "compression_sensitivity_table.tex")

    stress = split_selection_stress(profiles, args.cache, seeds, repeats=max(3, args.repeats // 2))
    stress.to_csv(out / "split_stress.csv", index=False)
    R.write_split_stress_table(stress, out / "split_stress_table.tex")

    if args.enable_learned_bottlenecks:
        bottleneck, bottleneck_profiles = learned_bottleneck_experiment(
            profiles, args.cache, args.data_root, seeds, repeats=args.repeats,
            bottleneck_channels=args.bottleneck_channels,
            bottleneck_epochs=args.bottleneck_epochs,
            alpha=args.alpha,
            raw_input_scale=args.bottleneck_raw_input_scale,
        )
        bottleneck.to_csv(out / "bottleneck.csv", index=False)
        (out / "bottleneck_profiles.json").write_text(json.dumps(bottleneck_profiles, indent=2), encoding="utf-8")
        R.write_bottleneck_table(bottleneck, out / "bottleneck_table.tex")

    R.make_figures(df, cal_df, out, displays)
    R.make_graphical_abstract(out)

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n=== method means (all datasets/seeds/profiles) ===")
    cols = ["accuracy", "mean_latency_ms", "p95_latency_ms", "deadline_miss_rate",
            "mean_upload_kb", "mean_energy_j", "exit_rate", "local_rate",
            "offload_rate", "selective_error"]
    print(df.groupby("method")[cols].mean().loc[R.MAIN_ORDER].round(4).to_string())
    print(f"\nwrote tables + figures to {out}")


if __name__ == "__main__":
    main()
