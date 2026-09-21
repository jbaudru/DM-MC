#!/usr/bin/env python3
"""
Evaluate baseline models (AKOM, CPT+, MOGen, IOHMM) on the next-node prediction task
and compare them with each other (and optionally with GraphIDyOMo results).

Usage
-----
    # Basic – evaluate all baselines on the test CSV
    python evaluate_baselines.py \
        --data-path  data/worldmove_380_NY_test.csv \
        --graph-path data/worldmove_380_NY_network.json

    # Full options
    python evaluate_baselines.py \
        --data-path          data/worldmove_380_NY_test.csv \
        --graph-path         data/worldmove_380_NY_network.json \
        --sequence-length    10 \
        --train-ratio        0.8 \
        --models             akom cpt mogen iohmm \
        --akom-max-order     10 \
        --mogen-max-order    5 \
        --iohmm-states       5 \
        --iohmm-max-iter     30 \
        --output-dir         results_baselines \
        --graphidyom-results experiments_all_configs   # optional comparison folder
"""

import argparse
import ast
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure UTF-8 on Windows
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Data loading helpers (mirrors train.py logic)
# ---------------------------------------------------------------------------

def load_trajectories(data_path: str,
                      sequence_length: int = 10) -> Tuple[List[List[int]], List[int]]:
    """
    Load and parse the CSV dataset.

    Returns
    -------
    full_paths : list of list of int  – all trajectories (raw node IDs)
    node_vocab : list of int          – sorted unique node IDs
    """
    df = pd.read_csv(data_path)

    if "q_path" in df.columns:
        def _parse(s):
            if isinstance(s, str):
                return [int(x.strip()) for x in s.split(",") if x.strip()]
            return []
        df["route_nodes"] = df["q_path"].apply(_parse)
    elif "route_taken" in df.columns:
        df["route_nodes"] = df["route_taken"].apply(ast.literal_eval)
    else:
        raise ValueError("Expected column 'q_path' or 'route_taken' in CSV.")

    df = df[df["route_nodes"].apply(len) >= sequence_length + 1]
    print(f"  Loaded {len(df)} trajectories (length >= {sequence_length + 1})")

    all_nodes = sorted({n for route in df["route_nodes"] for n in route})
    full_paths = df["route_nodes"].tolist()

    return full_paths, all_nodes


def build_adjacency(graph_path: str, node_vocab: List[int],
                    node_to_idx: Dict[int, int]) -> np.ndarray:
    """
    Build a binary adjacency matrix (V × V) aligned with node_to_idx.
    """
    V = len(node_vocab)
    adj = np.zeros((V, V), dtype=np.float32)

    with open(graph_path, "r") as f:
        graph_data = json.load(f)

    edges_key = "links" if "links" in graph_data else "edges"
    for edge in graph_data.get(edges_key, []):
        src = edge.get("source")
        tgt = edge.get("target")
        if src in node_to_idx and tgt in node_to_idx:
            adj[node_to_idx[src], node_to_idx[tgt]] = 1.0

    return adj


def encode_paths(full_paths: List[List[int]],
                 node_to_idx: Dict[int, int]) -> List[List[int]]:
    """Remap raw node IDs to dense integer indices."""
    encoded = []
    for path in full_paths:
        enc = [node_to_idx[n] for n in path if n in node_to_idx]
        if enc:
            encoded.append(enc)
    return encoded


def make_samples(encoded_paths: List[List[int]],
                 sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window extraction.

    Returns
    -------
    X : (N, sequence_length)  – context windows
    y : (N,)                  – next-node labels
    """
    X, y = [], []
    for path in encoded_paths:
        for i in range(len(path) - sequence_length):
            X.append(path[i: i + sequence_length])
            y.append(path[i + sequence_length])
    return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   vocab_size: int, desc: str = "") -> Dict:
    """
    Evaluate a fitted baseline model on a test set.

    Returns dict with accuracy, top-3, top-5 and mean reciprocal rank.
    """
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    rr_sum = 0.0
    n = len(y_test)

    t0 = time.time()
    for i in range(n):
        ctx = X_test[i].tolist()
        true_next = int(y_test[i])

        proba = model.predict_proba(ctx, vocab_size=vocab_size)
        ranked = np.argsort(proba)[::-1]

        if ranked[0] == true_next:
            correct_top1 += 1
        if true_next in ranked[:3]:
            correct_top3 += 1
        if true_next in ranked[:5]:
            correct_top5 += 1

        rank_pos = np.where(ranked == true_next)[0]
        if len(rank_pos) > 0:
            rr_sum += 1.0 / (rank_pos[0] + 1)

    elapsed = time.time() - t0

    results = {
        "accuracy_top1": correct_top1 / n,
        "accuracy_top3": correct_top3 / n,
        "accuracy_top5": correct_top5 / n,
        "mrr":           rr_sum / n,
        "n_samples":     n,
        "eval_time_s":   elapsed,
    }
    label = f"[{desc}]" if desc else ""
    print(f"  {label}  Acc@1={results['accuracy_top1']:.4f}  "
          f"Acc@3={results['accuracy_top3']:.4f}  "
          f"Acc@5={results['accuracy_top5']:.4f}  "
          f"MRR={results['mrr']:.4f}  "
          f"({n} samples, {elapsed:.1f}s)")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(results: Dict[str, Dict], output_dir: str,
                    graphidyom_results: Optional[Dict] = None) -> None:
    """Create a bar-chart comparison of all baseline accuracies."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
    except ImportError:
        print("  matplotlib not available – skipping plot.")
        return

    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
    })

    all_results = dict(results)
    if graphidyom_results:
        all_results.update(graphidyom_results)

    models_list = list(all_results.keys())
    metrics = ["accuracy_top1", "accuracy_top3", "accuracy_top5", "mrr"]
    labels  = ["Acc@1", "Acc@3", "Acc@5", "MRR"]
    colors  = ["#4ECDC4", "#45B7D1", "#FFA07A", "#FF6B6B"]

    x = np.arange(len(models_list))
    width = 0.20
    fig, ax = plt.subplots(figsize=(max(6, len(models_list) * 1.5), 4))

    for j, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [all_results[m].get(metric, 0.0) for m in models_list]
        offset = (j - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=color,
                      edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=6, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(models_list, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Next-Node Prediction – Baseline Comparison")
    ax.legend(loc="upper right", framealpha=0.8)
    ax.set_ylim(0, min(1.05, ax.get_ylim()[1] + 0.1))
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.7)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"baseline_comparison_{timestamp}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"\n  Plot saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate baseline models (AKOM, CPT+, MOGen, IOHMM).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---- shared params (mirror train_all_configs_ltm_stm.py) ---------------
    p.add_argument("--data-path",  default="data/worldmove_380_NY_test.csv",
                   help="Path to trajectory CSV.")
    p.add_argument("--graph-path", default="data/worldmove_380_NY_network.json",
                   help="Path to road-network JSON.")
    p.add_argument("--output-base", default="results_baselines",
                   help="Base output directory (matches --output-base in train script).")
    p.add_argument("--epochs-ltm", type=int, default=10,
                   help="Training epochs / EM iterations for iterative baselines "
                        "(IOHMM). Mirrors --epochs-ltm in train script.")
    p.add_argument("--epochs-stm", type=int, default=10,
                   help="Accepted for interface parity with train script "
                        "(not used by static baselines).")
    p.add_argument("--batch-size-ltm", type=int, default=128,
                   help="Accepted for interface parity with train script "
                        "(not used by static baselines).")
    p.add_argument("--batch-size-stm", type=int, default=32,
                   help="Accepted for interface parity with train script "
                        "(not used by static baselines).")
    # ---- evaluation-specific params ----------------------------------------
    p.add_argument("--sequence-length", type=int, default=10,
                   help="Context window length for next-node prediction.")
    p.add_argument("--train-ratio", type=float, default=0.8,
                   help="Fraction of trajectories used for training.")
    p.add_argument("--models", nargs="+",
                   default=["akom", "cpt", "mogen", "iohmm"],
                   choices=["akom", "cpt", "mogen", "iohmm"],
                   help="Which baselines to evaluate.")
    # AKOM
    p.add_argument("--akom-max-order", type=int, default=10,
                   help="Maximum Markov order for AKOM.")
    # MOGen
    p.add_argument("--mogen-max-order", type=int, default=5,
                   help="Maximum Markov order for MOGen.")
    p.add_argument("--mogen-significance", type=float, default=1e-3,
                   help="Likelihood-ratio test significance threshold for MOGen order selection.")
    # IOHMM  (max-iter is an alias kept for backward compat; --epochs-ltm takes precedence)
    p.add_argument("--iohmm-states", type=int, default=5,
                   help="Number of hidden states for IOHMM.")
    p.add_argument("--iohmm-max-iter", type=int, default=None,
                   help="Override EM iterations for IOHMM (default: use --epochs-ltm).")
    # CPT+
    p.add_argument("--cpt-max-context", type=int, default=5,
                   help="Maximum context length for CPT+.")
    # Comparison / misc
    p.add_argument("--graphidyom-results", default=None,
                   help="Optional: path to a GraphIDyOM experiments folder to "
                        "include in the comparison plot.")
    p.add_argument("--no-adjacency", action="store_true",
                   help="Disable topology masking (ignore road network structure).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_base, exist_ok=True)

    # IOHMM iterations: explicit override wins, otherwise fall back to --epochs-ltm
    iohmm_max_iter = args.iohmm_max_iter if args.iohmm_max_iter is not None else args.epochs_ltm

    # ------------------------------------------------------------------ data
    print("\n[1/4] Loading data …")
    full_paths, node_vocab = load_trajectories(args.data_path, args.sequence_length)

    node_to_idx = {n: i for i, n in enumerate(node_vocab)}
    vocab_size = len(node_vocab)
    print(f"  Vocabulary size: {vocab_size}")

    encoded_paths = encode_paths(full_paths, node_to_idx)

    # Train/test split (by trajectory, not sample, to avoid leakage)
    n_train = max(1, int(len(encoded_paths) * args.train_ratio))
    train_paths = encoded_paths[:n_train]
    test_paths  = encoded_paths[n_train:]
    print(f"  Train trajectories: {len(train_paths)}  |  "
          f"Test trajectories: {len(test_paths)}")

    X_test, y_test = make_samples(test_paths, args.sequence_length)
    print(f"  Test samples: {len(X_test)}")

    # ----------------------------------------------------------- adjacency
    print("\n[2/4] Building adjacency matrix …")
    if args.no_adjacency or not os.path.isfile(args.graph_path):
        adjacency = None
        print("  Topology masking disabled.")
    else:
        adjacency = build_adjacency(args.graph_path, node_vocab, node_to_idx)
        edge_count = int(adjacency.sum())
        print(f"  Adjacency matrix: {vocab_size}×{vocab_size}, {edge_count} edges")

    # ----------------------------------------------------------- models
    from models.baselines import AKOM, CPTPlus, MOGen, IOHMM

    model_registry = {
        "akom":  lambda: AKOM(max_order=args.akom_max_order),
        "cpt":   lambda: CPTPlus(max_context_length=args.cpt_max_context),
        "mogen": lambda: MOGen(max_order=args.mogen_max_order,
                               significance=args.mogen_significance),
        "iohmm": lambda: IOHMM(n_states=args.iohmm_states,
                                max_iter=iohmm_max_iter),
    }

    results: Dict[str, Dict] = {}

    print("\n[3/4] Training & evaluating baselines …")
    for model_name in args.models:
        print(f"\n  --- {model_name.upper()} ---")
        model = model_registry[model_name]()

        t0 = time.time()
        model.fit(train_paths, adjacency=adjacency)
        train_time = time.time() - t0
        print(f"  Fitted in {train_time:.1f}s")

        res = evaluate_model(model, X_test, y_test, vocab_size,
                             desc=model_name.upper())
        res["train_time_s"] = train_time
        results[model_name.upper()] = res

    # ----------------------------------------------------------- save JSON
    print("\n[4/4] Saving results …")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(args.output_base, f"baseline_results_{timestamp}.json")
    summary = {
        "timestamp": timestamp,
        "data_path":  args.data_path,
        "graph_path": args.graph_path,
        "sequence_length": args.sequence_length,
        "vocab_size": vocab_size,
        "n_train_trajectories": len(train_paths),
        "n_test_trajectories":  len(test_paths),
        "n_test_samples": len(X_test),
        "results": results,
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results JSON → {out_json}")

    # ----------------------------------------------------------- compare with GraphIDyOMo
    graphidyom_comparison: Optional[Dict] = None
    if args.graphidyom_results and os.path.isdir(args.graphidyom_results):
        graphidyom_comparison = _load_graphidyom_results(args.graphidyom_results)

    # ----------------------------------------------------------- plot
    plot_comparison(results, args.output_base, graphidyom_comparison)

    # ----------------------------------------------------------- summary table
    print("\n" + "=" * 72)
    print(f"{'Model':<15} {'Acc@1':>8} {'Acc@3':>8} {'Acc@5':>8} {'MRR':>8}")
    print("-" * 72)
    all_for_table = dict(results)
    if graphidyom_comparison:
        all_for_table.update(graphidyom_comparison)
    for name, r in sorted(all_for_table.items()):
        print(f"{name:<15} "
              f"{r.get('accuracy_top1', float('nan')):>8.4f} "
              f"{r.get('accuracy_top3', float('nan')):>8.4f} "
              f"{r.get('accuracy_top5', float('nan')):>8.4f} "
              f"{r.get('mrr', float('nan')):>8.4f}")
    print("=" * 72)


def _load_graphidyom_results(experiments_dir: str) -> Dict[str, Dict]:
    """
    Collect accuracy values from GraphIDyOMo experiment folders so they can be
    shown alongside baseline results in the comparison plot.

    Looks for JSON files produced by train_all_configs_ltm_stm.py.
    """
    results: Dict[str, Dict] = {}
    for folder in sorted(os.listdir(experiments_dir)):
        folder_path = os.path.join(experiments_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        # Walk sub-dirs looking for training_results.json
        for root, dirs, files in os.walk(folder_path):
            for fname in files:
                if fname == "training_results.json":
                    try:
                        with open(os.path.join(root, fname)) as f:
                            data = json.load(f)
                        acc = (data.get("results", {})
                                   .get("graphidyom", {})
                                   .get("accuracy"))
                        if acc is not None:
                            label = f"GraphIDyOMo\n({folder[:20]})"
                            results[label] = {
                                "accuracy_top1": float(acc),
                                "accuracy_top3": float(acc),
                                "accuracy_top5": float(acc),
                                "mrr": float(acc),
                            }
                    except Exception:
                        pass
    return results


if __name__ == "__main__":
    main()
