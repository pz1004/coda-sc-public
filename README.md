# CODA-SC: Calibrated Selective Edge-Cloud Inference Under Internet Deadlines

This repository contains the manuscript package and reproducible experimental
prototype for reliable split computing over variable Internet links, built on a
real multi-exit convolutional network with measured FLOPs/activation payloads
and documented device/network models.

## Artifacts

- `main.tex`, `main.pdf`, `supplement.tex`, `supplement.pdf`: manuscript files.
- `references.bib`: bibliography.
- `cover_letter.md`, `highlights.md`: submission-side text artifacts.
- `src/coda_cnn.py`: multi-exit CNN, training, and FLOPs-based profile measurement.
- `src/policy_eval.py`: CODA-SC-Cov/Risk exits, the coupled online controller, and all baselines.
- `src/coda_sc.py`: shared action/trace utilities plus a legacy sklearn prototype kept for archival comparison.
- `src/report.py`: statistics (CIs, Wilcoxon), LaTeX tables, and figures.
- `scripts/run_experiments.py`: single entry point for the whole pipeline.
- `results/`: generated CSVs, LaTeX tables, figures, profiles, and cached model outputs.

## Reproducing Experiments

```bash
python3 scripts/run_experiments.py --seeds 7,13,29,42,101 --repeats 6 \
  --enable-learned-bottlenecks --bottleneck-channels 8 \
  --bottleneck-epochs 8 --output results
```

This trains a multi-exit VGG-style CNN with three early-exit heads on four real
vision workloads (MNIST, Fashion-MNIST, CIFAR-10, SVHN) on a GPU, then measures
model FLOPs and exact int8 activation payloads. Compute latency/energy are
derived from those measurements plus documented edge/cloud device assumptions,
and the network layer is evaluated with modeled Wi-Fi/LTE/congested/mixed
bandwidth/RTT/queue/loss traces. It calibrates CODA-SC-Cov and CODA-SC-Risk,
compares against local, cloud, static split, oracle split, confidence-exit,
BranchyNet, deadline-greedy, an AODPart-style online-partitioning reference,
conformal-exit, and CRC-exit ablations. It writes CSV summaries, LaTeX tables
(with 95% CIs and paired Wilcoxon tests), calibration/action diagnostics,
sensitivity sweeps, plots, a learned-bottleneck split-selection table, and a
graphical abstract to `results/`. Trained model outputs and bottleneck codecs
are cached in `results/cnn_cache/`, so re-runs skip training.

Requires `torch`/`torchvision` (see `requirements.txt`) and a CUDA GPU for
training; policy evaluation runs on the cached logits.
