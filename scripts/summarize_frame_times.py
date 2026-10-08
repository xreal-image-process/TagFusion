#!/usr/bin/env python3
"""Summarize per-frame frontend / backend timings from FrameTimeStats.csv.

Offline ORB-SLAM3 runs LocalMapping and LoopClosing inside Track*(), so
FrameTimeStats.csv is written to the process cwd at Shutdown.

Usage:
  python3 scripts/summarize_frame_times.py
  python3 scripts/summarize_frame_times.py --csv path/to/FrameTimeStats.csv
  python3 scripts/summarize_frame_times.py --csv FrameTimeStats.csv --html times.html
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


STATE_NAME = {
    -1: "SYSTEM_NOT_READY",
    0: "NO_IMAGES_YET",
    1: "NOT_INITIALIZED",
    2: "OK",
    3: "RECENTLY_LOST",
    4: "LOST",
    5: "OK_KLT",
}

COLS = (
    "frontend_ms",
    "tag_detect_ms",
    "tag_ippe_ms",
    "tag_ms",
    "frontend_rest_ms",
    "local_mapping_ms",
    "loop_closing_ms",
    "backend_ms",
    "total_ms",
)

CSV_REQUIRED = (
    "frontend_ms",
    "local_mapping_ms",
    "loop_closing_ms",
    "backend_ms",
    "total_ms",
)


def percentile(xs: np.ndarray, q: float) -> float:
    if xs.size == 0:
        return float("nan")
    return float(np.percentile(xs, q))


def summarize(xs: np.ndarray) -> dict[str, float]:
    if xs.size == 0:
        return {k: float("nan") for k in ("n", "mean", "std", "min", "p50", "p90", "p99", "max")}
    return {
        "n": float(xs.size),
        "mean": float(xs.mean()),
        "std": float(xs.std()),
        "min": float(xs.min()),
        "p50": percentile(xs, 50),
        "p90": percentile(xs, 90),
        "p99": percentile(xs, 99),
        "max": float(xs.max()),
    }


def load_csv(path: Path) -> dict[str, np.ndarray]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise SystemExit(f"empty csv: {path}")

    def col(name: str, dtype=float) -> np.ndarray:
        return np.array([dtype(r[name]) for r in rows])

    def optional_col(name: str) -> np.ndarray:
        if name not in rows[0] or rows[0][name] == "":
            return np.zeros(len(rows), dtype=float)
        return np.array([float(r[name]) if r[name] != "" else 0.0 for r in rows])

    frontend = col("frontend_ms")
    tag_detect = optional_col("tag_detect_ms")
    tag_ippe = optional_col("tag_ippe_ms")
    tag = tag_detect + tag_ippe
    return {
        "frame_id": col("frame_id", int),
        "timestamp": col("timestamp"),
        "state": col("state", int),
        "is_kf": col("is_kf", int),
        "frontend_ms": frontend,
        "tag_detect_ms": tag_detect,
        "tag_ippe_ms": tag_ippe,
        "tag_ms": tag,
        "frontend_rest_ms": frontend - tag,
        **{name: col(name) for name in CSV_REQUIRED if name != "frontend_ms"},
    }


def print_block(title: str, data: dict[str, np.ndarray], mask: np.ndarray | None = None) -> None:
    if mask is None:
        mask = np.ones(data["frame_id"].shape[0], dtype=bool)
    n = int(mask.sum())
    print(f"\n== {title}  n={n} ==")
    if n == 0:
        return
    for name in COLS:
        s = summarize(data[name][mask])
        print(f"{name:<18} mean={s['mean']:.2f} ms")


def maybe_html(data: dict[str, np.ndarray], path: Path) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly not installed; skip HTML")
        return

    t = data["timestamp"]
    t0 = float(t[0]) if t.size else 0.0
    rel = t - t0
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("per-frame time [ms]", "backend split [ms]"))
    fig.add_trace(go.Scatter(x=rel, y=data["frontend_ms"], name="frontend",
                             line=dict(color="#4C72B0", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=rel, y=data["tag_detect_ms"], name="tag detect",
                             line=dict(color="#17BECF", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=rel, y=data["tag_ippe_ms"], name="tag ippe",
                             line=dict(color="#BCBD22", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=rel, y=data["backend_ms"], name="backend",
                             line=dict(color="#D62728", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=rel, y=data["total_ms"], name="total",
                             line=dict(color="#333333", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=rel, y=data["local_mapping_ms"], name="local mapping",
                             line=dict(color="#2CA02C", width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=rel, y=data["loop_closing_ms"], name="loop closing",
                             line=dict(color="#9467BD", width=1)), row=2, col=1)
    kf = data["is_kf"].astype(bool)
    if kf.any():
        fig.add_trace(
            go.Scatter(
                x=rel[kf],
                y=data["total_ms"][kf],
                mode="markers",
                name="keyframe",
                marker=dict(size=5, color="#FF7F0E"),
            ),
            row=1,
            col=1,
        )
    fig.update_xaxes(title_text="time since first frame [s]", row=2, col=1)
    fig.update_yaxes(title_text="ms", row=1, col=1)
    fig.update_yaxes(title_text="ms", row=2, col=1)
    fig.update_layout(width=1100, height=720, title=str(path.stem))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs=True, full_html=True)
    print(f"saved HTML -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize FrameTimeStats.csv")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("FrameTimeStats.csv"),
        help="input CSV (default: ./FrameTimeStats.csv)",
    )
    parser.add_argument("--html", type=Path, default=None, help="optional Plotly HTML")
    parser.add_argument("--ok-only", action="store_true", help="only tracking-OK frames")
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        print(f"csv not found: {csv_path}")
        print("Rebuild, re-run the sequence, then point --csv at FrameTimeStats.csv in the run cwd.")
        return 1

    data = load_csv(csv_path)
    n = data["frame_id"].size
    n_kf = int(data["is_kf"].sum())
    states, counts = np.unique(data["state"], return_counts=True)
    print(f"csv: {csv_path}")
    print(f"frames={n}  keyframes={n_kf}  non-kf={n - n_kf}")
    print("tracking state:")
    for s, c in zip(states, counts):
        print(f"  {int(s)} {STATE_NAME.get(int(s), '?')}: {int(c)}")

    mask = np.ones(n, dtype=bool)
    if args.ok_only:
        mask = data["state"] == 2

    print_block("all frames", data, mask)
    print_block("keyframes only", data, mask & (data["is_kf"] == 1))
    print_block("non-keyframes", data, mask & (data["is_kf"] == 0))

    if args.html is not None:
        maybe_html(data, args.html.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
