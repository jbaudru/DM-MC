#!/usr/bin/env python3
"""
plot_efficiency.py – Memory and inference-time profiling for all models.

Produces publication-quality figures that support the lightweight / real-time
deployment claim for DM-MC (GraphIDyOM categorical_id).

Outputs (written to --output-dir)
----------------------------------
  efficiency_inference_time.pdf/.png  – Inference time (ms / prediction)
  efficiency_memory_disk.pdf/.png     – On-disk model size (MB)
  efficiency_memory_ram.pdf/.png      – Peak in-RAM footprint (MB)
  efficiency_combined.pdf/.png        – Side-by-side 3-panel figure for the paper

Usage
-----
    python plot_efficiency.py \\
        --models-dir saved_models \\
        --output-dir results_unified \\
        --n-warmup 20 \\
        --n-repeats 200

The script reads the same split-0 test windows used by evaluate_models.py
(from test_paths.pkl of the first dataset / split it finds) to ensure the
inference-time sample is representative of real inference inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys
import time
import tracemalloc
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch

from evaluate_models import (
    load_graphidyom,
    load_baseline,
    make_graphidyom_proba_fn,
    make_baseline_proba_fn,
)
from train_and_evaluate import make_samples_with_users
from plot_results import _display_name

# ---------------------------------------------------------------------------
# Colour / style constants (IEEE palette, colour-blind safe)
# ---------------------------------------------------------------------------
_COLOR_PROPOSED  = "#E63946"   # vivid red  → DM-MC (our model)
_COLOR_COMPLEX   = "#457B9D"   # steel blue → AKOM, CPT, MOGen, IOHMM
_COLOR_MARKOV    = "#2A9D8F"   # teal       → simple Markov (MC*p, HW-MC*)
_COLOR_GE        = "#A8DADC"   # light cyan → GE-encoded variants (secondary)


def _model_color(name: str) -> str:
    up = name.upper()
    if "GRAPHIDYOM_CATEGORICAL_ID" in up or name.startswith("GraphIDyOM_categorical_id"):
        return _COLOR_PROPOSED
    if up in {"AKOM", "CPT", "MOGEN", "IOHMM"}:
        return _COLOR_COMPLEX
    return _COLOR_MARKOV


def _model_hatch(name: str) -> str:
    """Distinct hatch per model family for monochrome readability."""
    up = name.upper()
    if "GRAPHIDYOM_CATEGORICAL_ID" in up:
        return ""
    if up in {"AKOM", "CPT", "MOGEN", "IOHMM"}:
        return "//"
    return ".."


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def discover_models(split_dir: str, orders: List[int],
                    encodings: List[str]) -> List[Dict]:
    """
    Enumerate model files in *split_dir*.

    Returns a list of dicts with keys:
        key        – unique identifier matching evaluate_models.py naming
        path       – absolute file path
        kind       – 'graphidyom' or 'baseline'
        encoding   – encoding string (graphidyom only, else '')
        order      – Markov order (graphidyom only, else 0)
        baseline   – baseline name string (baseline only, else '')
    """
    models = []

    # GraphIDyOM checkpoints
    for enc in encodings:
        for order in orders:
            fname = f"graphidyom_{enc}_ord{order}.pt"
            fpath = os.path.join(split_dir, fname)
            if os.path.isfile(fpath):
                models.append(dict(
                    key=f"GraphIDyOM_{enc}_ord{order}",
                    path=fpath,
                    kind="graphidyom",
                    encoding=enc,
                    order=order,
                    baseline="",
                ))

    # Baseline pickles
    for entry in os.listdir(split_dir):
        if entry.startswith("baseline_") and entry.endswith(".pkl"):
            bname = entry[len("baseline_"):-len(".pkl")]
            models.append(dict(
                key=bname.upper(),
                path=os.path.join(split_dir, entry),
                kind="baseline",
                encoding="",
                order=0,
                baseline=bname,
            ))

    return models


# ---------------------------------------------------------------------------
# Inference-time benchmark
# ---------------------------------------------------------------------------

def benchmark_inference(
    model_info: Dict,
    test_contexts: np.ndarray,
    vocab_size: int,
    device: torch.device,
    n_warmup: int = 20,
    n_repeats: int = 200,
) -> Dict:
    """
    Return median and mean inference time (ms / prediction) for *model_info*.

    Uses the first ``n_warmup + n_repeats`` rows of *test_contexts*
    (cycling if the array is shorter).

    Returns a dict with: median_ms, mean_ms, std_ms, p95_ms
    """
    n_total = n_warmup + n_repeats
    ctx_idx = np.arange(n_total) % len(test_contexts)
    contexts = test_contexts[ctx_idx]  # (n_total, seq_len)

    if model_info["kind"] == "graphidyom":
        ckpt_model, graph_structure, _vs, _ = load_graphidyom(
            model_info["path"], device)
        fn = make_graphidyom_proba_fn(ckpt_model, graph_structure, device)
    else:
        bl_model, _vs, _ = load_baseline(model_info["path"])
        fn = make_baseline_proba_fn(bl_model, vocab_size)

    # Warm-up
    for i in range(n_warmup):
        _ = fn(contexts[i].tolist())

    # Timed runs
    times_ms = []
    for i in range(n_warmup, n_total):
        t0 = time.perf_counter()
        _ = fn(contexts[i].tolist())
        times_ms.append((time.perf_counter() - t0) * 1e3)

    times_ms = np.array(times_ms)
    return {
        "median_ms": float(np.median(times_ms)),
        "mean_ms":   float(np.mean(times_ms)),
        "std_ms":    float(np.std(times_ms)),
        "p95_ms":    float(np.percentile(times_ms, 95)),
    }


# ---------------------------------------------------------------------------
# Memory profiling
# ---------------------------------------------------------------------------

def measure_disk_mb(model_info: Dict) -> float:
    """Return on-disk file size in MB."""
    try:
        return os.path.getsize(model_info["path"]) / (1024 ** 2)
    except OSError:
        return float("nan")


def measure_ram_mb(model_info: Dict, device: torch.device) -> float:
    """
    Peak additional RAM (MB) allocated while loading the model.

    Uses ``tracemalloc`` to capture the Python-heap allocation delta during
    model loading.  For GPU-placed GraphIDyOM models the matrix data
    resides on the GPU; the figure therefore reflects the CPU-side footprint
    of the graph-structure dicts and model object—an important metric because
    real-time deployment typically shares host memory with other services.
    """
    tracemalloc.start()
    snap0 = tracemalloc.take_snapshot()

    try:
        if model_info["kind"] == "graphidyom":
            ckpt_model, graph_structure, _vs, _ = load_graphidyom(
                model_info["path"], device)
            # Keep references alive until after snapshot so GC doesn't skew
            _ref = (ckpt_model, graph_structure)
        else:
            bl_model, _vs, _ = load_baseline(model_info["path"])
            _ref = bl_model

        snap1 = tracemalloc.take_snapshot()
        stats = snap1.compare_to(snap0, "lineno")
        delta_bytes = sum(s.size_diff for s in stats if s.size_diff > 0)
    finally:
        tracemalloc.stop()

    return delta_bytes / (1024 ** 2)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         9,
        "axes.labelsize":    10,
        "axes.titlesize":    10,
        "legend.fontsize":   8,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "axes.grid":         True,
        "grid.alpha":        0.35,
        "grid.linestyle":    "--",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })
    return plt, ticker


def _safe_values(values: List[float]) -> Tuple[List[float], bool]:
    """Replace NaN/0 with a small epsilon; return (cleaned, any_positive)."""
    EPS = 1e-6
    cleaned = [max(v, EPS) if math.isfinite(v) and v > 0
               else (EPS if not math.isfinite(v) or v <= 0 else v)
               for v in values]
    any_pos = any(v > EPS * 2 for v in values if math.isfinite(v))
    return cleaned, any_pos


def _bar_plot(
    plt,
    names: List[str],
    values: List[float],
    errors: Optional[List[float]],
    colors: List[str],
    hatches: List[str],
    ylabel: str,
    title: str,
    log_scale: bool = False,
) -> "plt.Figure":
    """Single horizontal-bar chart with optional error bars."""
    safe_vals, any_pos = _safe_values(values)
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.4 * len(names))))
    y_pos = np.arange(len(names))
    ax.barh(
        y_pos, safe_vals,
        xerr=errors,
        color=colors,
        hatch=hatches,
        edgecolor="white",
        linewidth=0.6,
        error_kw=dict(ecolor="#555", capsize=3, linewidth=0.8),
        height=0.65,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(ylabel)
    ax.set_title(title, pad=6)
    if log_scale and any_pos:
        ax.set_xscale("log")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def _combined_plot(
    plt,
    names: List[str],
    inf_median: List[float],
    inf_std: List[float],
    disk_mb: List[float],
    ram_mb: List[float],
    colors: List[str],
    hatches: List[str],
) -> "plt.Figure":
    """Two- or three-panel figure: inference time | disk | RAM (RAM omitted if all NaN)."""
    ram_any = any(math.isfinite(v) and v > 0 for v in ram_mb)
    n_panels = 3 if ram_any else 2
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(4.8 * n_panels, max(3.5, 0.38 * len(names))))
    if n_panels == 2:
        axes = list(axes)
    else:
        axes = list(axes)

    all_panels = [
        (inf_median, inf_std,  "Inference time (ms / prediction)",    "Latency"),
        (disk_mb,    None,     "On-disk model size (MB)",              "Disk footprint"),
        (ram_mb,     None,     "Peak RAM during load (MB)",            "RAM footprint"),
    ]
    panels = all_panels[:n_panels]

    y_pos = np.arange(len(names))
    for ax_idx, (ax, (vals, errs, xlabel, title)) in enumerate(zip(axes, panels)):
        safe_vals, any_pos = _safe_values(vals)
        ax.barh(
            y_pos, safe_vals,
            xerr=errs,
            color=colors,
            hatch=hatches,
            edgecolor="white",
            linewidth=0.6,
            error_kw=dict(ecolor="#555", capsize=2, linewidth=0.7),
            height=0.65,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names if ax_idx == 0 else [""] * len(names), fontsize=8)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=10, pad=5)
        if any_pos:
            ax.set_xscale("log")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.35, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=_COLOR_PROPOSED, label="DM-MC (proposed)"),
        Patch(facecolor=_COLOR_COMPLEX,  hatch="//", label="Seq. baselines (AKOM/CPT/MOGen/IOHMM)"),
        Patch(facecolor=_COLOR_MARKOV,   hatch="..", label="Markov baselines (MC*)"),
    ]
    fig.legend(handles=legend_items, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.03), fontsize=8,
               framealpha=0.9)
    fig.suptitle("Computational efficiency: inference latency and memory footprint",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Ordering / display names (mirror evaluate_models / plot_results conventions)
# ---------------------------------------------------------------------------

def _mc_baseline_type_key(up: str) -> int:
    """Group simple-Markov baselines by type so HW-MC* and MC* stay contiguous.

    0 = entropy-weighted (HW-MCk: mc2/mc5/mc10)
    1 = pure-backoff      (MCkP:   mc2p/mc5p/mc10p)
    2 = graph-embedding   (MCk_GE: mc2_ge/mc5_ge/mc10_ge)
    """
    if up.endswith("_GE"):
        return 2
    if re.fullmatch(r"MC\d+P", up):
        return 1
    return 0


def _mc_baseline_order_key(up: str) -> int:
    """Extract the Markov order from a baseline key (MC10P -> 10, MC5 -> 5)."""
    m = re.fullmatch(r"MC(\d+)(?:P|_GE)?", up)
    return int(m.group(1)) if m else 0


def _ordered_model_keys(infos: List[Dict]) -> List[Dict]:
    """
    Sort models: DM-MC (cat-id) first, then complex baselines, then MC* variants
    (HW-MC* grouped together, then plain MC* grouped together), then other
    GraphIDyOM encodings (GE, De Bruijn).
    Within each group: ascending Markov order.
    """
    def _sort_key(info: Dict):
        up = info["key"].upper()
        if info["kind"] == "graphidyom" and info["encoding"] == "categorical_id":
            return (0, info["order"], info["key"])
        if up in {"AKOM", "CPT", "MOGEN", "IOHMM"}:
            return (1, 0, 0, up)
        if info["kind"] == "baseline":
            return (2, _mc_baseline_type_key(up), _mc_baseline_order_key(up), up)
        # other GraphIDyOM encodings
        return (3, 0, info["order"], info["key"])
    return sorted(infos, key=_sort_key)


def _display(key: str) -> str:
    """Human-readable label used in plots."""
    return _display_name(key)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Profile inference time and memory for all saved models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models-dir",  default="saved_models")
    p.add_argument("--output-dir",  default="results_unified")
    p.add_argument("--n-warmup",    type=int, default=20,
                   help="Number of warm-up predictions before timing.")
    p.add_argument("--n-repeats",   type=int, default=200,
                   help="Number of timed predictions per model.")
    p.add_argument("--dataset-idx", type=int, default=0,
                   help="Which dataset index (0-based) to use for test contexts.")
    p.add_argument("--split-idx",   type=int, default=0,
                   help="Which split index to use for test contexts.")
    p.add_argument("--no-ram",      action="store_true",
                   help="Skip tracemalloc RAM measurement (faster, but panel omitted).")
    p.add_argument("--skip-encodings", nargs="*",
                   default=["debruijn", "graph_embedding"],
                   help="Skip these GraphIDyOM encodings to keep the figure focused "
                        "on categorical-id (node-native) models. Pass '' to include all.")
    return p.parse_args()


def main():
    args = parse_args()
    plt, ticker = _setup_matplotlib()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cpu")   # CPU timing for fair cross-model comparison
    print(f"Device: {device}  (CPU used for fair latency comparison)")

    # ---- Load metadata -------------------------------------------------------
    meta_path = os.path.join(args.models_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        print(f"ERROR: {meta_path} not found.  Run train_models.py first.")
        return 1
    with open(meta_path) as f:
        meta = json.load(f)

    orders    = meta.get("orders",    [2, 5, 10])
    encodings = meta.get("encodings", ["categorical_id"])
    seq_len   = meta.get("sequence_length", 10)

    # ---- Locate split directory ----------------------------------------------
    datasets = meta.get("datasets", [])
    if not datasets:
        print("ERROR: no datasets found in metadata.json")
        return 1

    ds_idx = min(args.dataset_idx, len(datasets) - 1)
    ds_info = datasets[ds_idx]
    splits  = ds_info.get("splits", [])
    split_info = next(
        (s for s in splits if s["split_idx"] == args.split_idx),
        splits[0] if splits else None,
    )
    if split_info is None:
        print("ERROR: split not found.")
        return 1

    split_dir    = split_info["split_dir"]
    vocab_size   = split_info["vocab_size_cat"]
    ds_label     = ds_info.get("label", "dataset")
    print(f"Using split: {split_dir}  (dataset={ds_label}, vocab={vocab_size})")

    # ---- Load test contexts --------------------------------------------------
    tp_path = os.path.join(split_dir, "test_paths.pkl")
    if not os.path.isfile(tp_path):
        print(f"ERROR: {tp_path} not found.")
        return 1
    with open(tp_path, "rb") as f:
        tp = pickle.load(f)

    cat_test       = tp["cat_test"]
    cat_test_users = tp.get("cat_test_users", list(range(len(cat_test))))
    X_test, y_test, _ = make_samples_with_users(cat_test, cat_test_users, seq_len)
    if len(X_test) == 0:
        print("ERROR: no test windows found.")
        return 1
    print(f"Test contexts: {len(X_test)} windows (seq_len={seq_len}, vocab={vocab_size})")

    # ---- Discover models -----------------------------------------------------
    skip_enc = set(args.skip_encodings or [])
    all_infos = discover_models(split_dir, orders, encodings)
    # Filter skip-encodings
    all_infos = [
        m for m in all_infos
        if not (m["kind"] == "graphidyom" and m["encoding"] in skip_enc)
    ]
    # Also filter mc*_ge baselines when graph_embedding is skipped
    if "graph_embedding" in skip_enc:
        all_infos = [m for m in all_infos
                     if not m["key"].upper().endswith("_GE")]

    all_infos = _ordered_model_keys(all_infos)
    print(f"\nFound {len(all_infos)} model(s):")
    for m in all_infos:
        print(f"  {m['key']:45s}  {os.path.basename(m['path'])}")

    # ---- Benchmark loop ------------------------------------------------------
    results = []
    for i, info in enumerate(all_infos):
        label = _display(info["key"])
        print(f"\n[{i+1}/{len(all_infos)}] {info['key']}  ({label})")

        disk_mb = measure_disk_mb(info)
        print(f"  disk: {disk_mb:.2f} MB")

        # RAM measurement
        ram_mb = float("nan")
        if not args.no_ram:
            try:
                ram_mb = measure_ram_mb(info, device)
                print(f"  RAM delta: {ram_mb:.2f} MB")
            except Exception as exc:
                print(f"  RAM measure failed: {exc}")

        # Inference time
        try:
            timing = benchmark_inference(
                info, X_test, vocab_size, device,
                n_warmup=args.n_warmup, n_repeats=args.n_repeats)
            print(f"  latency: median={timing['median_ms']:.3f}ms  "
                  f"mean={timing['mean_ms']:.3f}ms  "
                  f"p95={timing['p95_ms']:.3f}ms")
        except Exception as exc:
            print(f"  latency measure failed: {exc}")
            timing = {"median_ms": float("nan"), "mean_ms": float("nan"),
                      "std_ms": float("nan"), "p95_ms": float("nan")}

        results.append({
            **info,
            "label":      label,
            "disk_mb":    disk_mb,
            "ram_mb":     ram_mb,
            "median_ms":  timing["median_ms"],
            "mean_ms":    timing["mean_ms"],
            "std_ms":     timing["std_ms"],
            "p95_ms":     timing["p95_ms"],
        })

    # ---- Save raw numbers as JSON for reproducibility -----------------------
    json_path = os.path.join(args.output_dir, "efficiency_results.json")
    _serialisable = []
    for r in results:
        _serialisable.append({k: (float(v) if isinstance(v, (np.floating, float)) else v)
                               for k, v in r.items()})
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_serialisable, f, indent=2)
    print(f"\nRaw numbers → {json_path}")

    # ---- Build plot arrays ---------------------------------------------------
    # Only include models with valid timing (skip degenerate NaN entries)
    valid = [r for r in results if math.isfinite(r["median_ms"])]
    if not valid:
        print("No valid timing results – aborting plot generation.")
        return 1

    names   = [r["label"] for r in valid]
    colors  = [_model_color(r["key"]) for r in valid]
    hatches = [_model_hatch(r["key"]) for r in valid]

    inf_med = [r["median_ms"] for r in valid]
    inf_std = [r["std_ms"]    for r in valid]
    disk    = [r["disk_mb"]   for r in valid]
    ram     = [r["ram_mb"]    for r in valid]

    # ---- Panel 1: inference time --------------------------------------------
    fig1 = _bar_plot(plt, names, inf_med, inf_std, colors, hatches,
                     ylabel="Inference time (ms / prediction)",
                     title="Inference latency (CPU, single prediction)",
                     log_scale=True)
    _save(fig1, plt, args.output_dir, "efficiency_inference_time")

    # ---- Panel 2: on-disk model size ----------------------------------------
    disk_valid = [v for v in disk if math.isfinite(v)]
    if disk_valid:
        fig2 = _bar_plot(plt, names, disk, None, colors, hatches,
                         ylabel="On-disk model size (MB)",
                         title="Model storage footprint",
                         log_scale=True)
        _save(fig2, plt, args.output_dir, "efficiency_memory_disk")

    # ---- Panel 3: RAM -------------------------------------------------------
    ram_valid = [v for v in ram if math.isfinite(v)]
    if ram_valid and not args.no_ram:
        fig3 = _bar_plot(plt, names, ram, None, colors, hatches,
                         ylabel="Peak RAM during load (MB)",
                         title="In-memory RAM footprint",
                         log_scale=True)
        _save(fig3, plt, args.output_dir, "efficiency_memory_ram")

    # ---- Combined 3-panel figure --------------------------------------------
    _ram_for_combined = ram if (ram_valid and not args.no_ram) else [float("nan")] * len(valid)
    fig4 = _combined_plot(plt, names, inf_med, inf_std, disk, _ram_for_combined,
                          colors, hatches)
    _save(fig4, plt, args.output_dir, "efficiency_combined")

    # ---- Print summary table ------------------------------------------------
    _print_table(valid)

    print(f"\n✓  Efficiency plots saved to: {args.output_dir}/")
    return 0


def _save(fig, plt, out_dir: str, stem: str) -> None:
    for ext in ("pdf", "png"):
        fpath = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(fpath)
        print(f"  → {fpath}")
    plt.close(fig)


def _print_table(results: List[Dict]) -> None:
    print("\n" + "=" * 85)
    print(f"{'Model':<35}  {'Latency med (ms)':>16}  {'Disk (MB)':>9}  {'RAM (MB)':>9}")
    print("-" * 85)
    for r in results:
        ram_str  = f"{r['ram_mb']:.2f}" if math.isfinite(r["ram_mb"]) else "  n/a "
        disk_str = f"{r['disk_mb']:.2f}" if math.isfinite(r["disk_mb"]) else "  n/a "
        print(f"{r['label']:<35}  {r['median_ms']:>16.4f}  {disk_str:>9}  {ram_str:>9}")
    print("=" * 85)


if __name__ == "__main__":
    sys.exit(main())
