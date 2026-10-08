#!/usr/bin/env python3
"""Evaluate TagMap coplanarity against a fitted plane GT + visualization.

Reads tag_map_corners.csv (TagMapExporter online/final export).

Pipeline:
  1) Per-tag center / normal from corners.
  2) Fit plane to tag centers (SVD) as GT.
  3) Report normal error (deg) and translation error (cm).
  4) Write interactive HTML: 3D + two bar charts.

Usage:
  python3 scripts/evaluate_tag_map_plane.py
  python3 scripts/evaluate_tag_map_plane.py --csv path/to/tag_map_corners.csv
  python3 scripts/evaluate_tag_map_plane.py --csv ... --html out.html --png out.png
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


SCRIPT_DIR = Path(__file__).resolve().parent
ORB_ROOT = SCRIPT_DIR.parent
DEFAULT_CSV = ORB_ROOT / "Examples/TagFusion/XREAL_Glass/tag_export/tag_map_corners.csv"


@dataclass
class TagGeom:
    tag_id: int
    is_fixed: bool
    corners: np.ndarray  # (4, 3)
    center: np.ndarray  # (3,)
    normal: np.ndarray  # (3,) unit


def load_tags(path: Path) -> list[TagGeom]:
    if not path.is_file():
        raise FileNotFoundError(f"missing tag map: {path}")

    tags: list[TagGeom] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row.get("has_corners", "0")) == 0:
                continue
            corners = np.array(
                [
                    [float(row["c0_x"]), float(row["c0_y"]), float(row["c0_z"])],
                    [float(row["c1_x"]), float(row["c1_y"]), float(row["c1_z"])],
                    [float(row["c2_x"]), float(row["c2_y"]), float(row["c2_z"])],
                    [float(row["c3_x"]), float(row["c3_y"]), float(row["c3_z"])],
                ],
                dtype=np.float64,
            )
            center = corners.mean(axis=0)
            e0 = corners[1] - corners[0]
            e1 = corners[3] - corners[0]
            n = np.cross(e0, e1)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            tags.append(
                TagGeom(
                    tag_id=int(row["tag_id"]),
                    is_fixed=int(row.get("is_fixed", "0")) != 0,
                    corners=corners,
                    center=center,
                    normal=n,
                )
            )
    return tags


def fit_plane(centers: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit plane n·x + d = 0 with ||n||=1 via SVD on centered points."""
    if centers.shape[0] < 3:
        raise ValueError("need at least 3 tag centers to fit a plane")
    centroid = centers.mean(axis=0)
    _, _, vt = np.linalg.svd(centers - centroid, full_matrices=False)
    n = vt[-1, :]
    n = n / np.linalg.norm(n)
    d = -float(np.dot(n, centroid))
    return n, d


def angle_deg(n0: np.ndarray, n1: np.ndarray) -> float:
    c = float(np.clip(np.abs(np.dot(n0, n1)), 0.0, 1.0))
    return math.degrees(math.acos(c))


def evaluate(tags: list[TagGeom]) -> dict:
    if len(tags) < 3:
        raise ValueError(f"need >= 3 tags with corners, got {len(tags)}")

    centers = np.stack([t.center for t in tags], axis=0)
    n_plane, d = fit_plane(centers)

    mean_n = np.mean(np.stack([t.normal for t in tags], axis=0), axis=0)
    if np.linalg.norm(mean_n) > 1e-12 and np.dot(n_plane, mean_n) < 0:
        n_plane = -n_plane
        d = -d

    rows = []
    for t in tags:
        n_tag = t.normal.copy()
        if np.dot(n_tag, n_plane) < 0:
            n_tag = -n_tag
        normal_err = angle_deg(n_tag, n_plane)
        trans_err_m = abs(float(np.dot(n_plane, t.center) + d))
        rows.append(
            {
                "tag_id": t.tag_id,
                "is_fixed": int(t.is_fixed),
                "normal_err_deg": normal_err,
                "trans_err_cm": trans_err_m * 100.0,
                "center_x": t.center[0],
                "center_y": t.center[1],
                "center_z": t.center[2],
                "nx": n_tag[0],
                "ny": n_tag[1],
                "nz": n_tag[2],
            }
        )

    normal_errs = np.array([r["normal_err_deg"] for r in rows], dtype=np.float64)
    trans_errs = np.array([r["trans_err_cm"] for r in rows], dtype=np.float64)
    return {
        "plane_n": n_plane,
        "plane_d": d,
        "rows": rows,
        "mean_normal_err_deg": float(normal_errs.mean()),
        "median_normal_err_deg": float(np.median(normal_errs)),
        "std_normal_err_deg": float(normal_errs.std()),
        "mean_trans_err_cm": float(trans_errs.mean()),
        "median_trans_err_cm": float(np.median(trans_errs)),
        "std_trans_err_cm": float(trans_errs.std()),
        "max_normal_err_deg": float(normal_errs.max()),
        "max_trans_err_cm": float(trans_errs.max()),
        "num_tags": len(rows),
    }


def write_report(path: Path, result: dict) -> None:
    """Write combined report plus two split CSVs (normal deg / translation cm)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tag_id",
                "is_fixed",
                "normal_err_deg",
                "trans_err_cm",
                "center_x",
                "center_y",
                "center_z",
                "nx",
                "ny",
                "nz",
            ],
        )
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow(row)

    normal_path = path.with_name(path.stem + "_normal.csv")
    trans_path = path.with_name(path.stem + "_translation.csv")
    with normal_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["tag_id", "is_fixed", "normal_err_deg"]
        )
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow(
                {
                    "tag_id": row["tag_id"],
                    "is_fixed": row["is_fixed"],
                    "normal_err_deg": row["normal_err_deg"],
                }
            )
        writer.writerow(
            {
                "tag_id": "MEAN",
                "is_fixed": "-",
                "normal_err_deg": result["mean_normal_err_deg"],
            }
        )
    with trans_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["tag_id", "is_fixed", "trans_err_cm"]
        )
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow(
                {
                    "tag_id": row["tag_id"],
                    "is_fixed": row["is_fixed"],
                    "trans_err_cm": row["trans_err_cm"],
                }
            )
        writer.writerow(
            {
                "tag_id": "MEAN",
                "is_fixed": "-",
                "trans_err_cm": result["mean_trans_err_cm"],
            }
        )
    print(f"wrote: {normal_path}")
    print(f"wrote: {trans_path}")


def _plane_mesh(
    n: np.ndarray, d: float, centers: np.ndarray, scale: float = 1.15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a rectangular patch of the plane covering tag centers."""
    # Orthonormal basis on plane
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    centroid = centers.mean(axis=0)
    # Project centroid onto plane
    centroid = centroid - (np.dot(n, centroid) + d) * n

    coords = centers - centroid
    su = coords @ u
    sv = coords @ v
    half_u = max(float(np.max(np.abs(su))), 0.05) * scale
    half_v = max(float(np.max(np.abs(sv))), 0.05) * scale

    corners = np.stack(
        [
            centroid + (-half_u) * u + (-half_v) * v,
            centroid + (half_u) * u + (-half_v) * v,
            centroid + (half_u) * u + (half_v) * v,
            centroid + (-half_u) * u + (half_v) * v,
        ],
        axis=0,
    )
    return corners[:, 0], corners[:, 1], corners[:, 2]


def build_figure(tags: list[TagGeom], result: dict) -> go.Figure:
    """3D TagMap+plane on top; two separate bar charts below (normal deg / trans cm)."""
    n_plane = result["plane_n"]
    d = result["plane_d"]
    row_by_id = {r["tag_id"]: r for r in result["rows"]}
    centers = np.stack([t.center for t in tags], axis=0)
    max_t_err_cm = max(r["trans_err_cm"] for r in result["rows"]) + 1e-12
    colorscale = "Turbo"

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "scene", "colspan": 2}, None],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        row_heights=[0.55, 0.45],
        subplot_titles=(
            "TagMap + GT plane (color = t_err [cm])",
            "Normal error [deg]",
            "Translation error [cm]",
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    px, py, pz = _plane_mesh(n_plane, d, centers)
    fig.add_trace(
        go.Mesh3d(
            x=px,
            y=py,
            z=pz,
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color="rgba(80,160,255,0.30)",
            name="GT plane",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    c0 = centers.mean(axis=0)
    c0 = c0 - (np.dot(n_plane, c0) + d) * n_plane
    arrow_len = max(0.08, 0.25 * float(np.linalg.norm(centers.max(0) - centers.min(0))))
    tip = c0 + n_plane * arrow_len
    fig.add_trace(
        go.Scatter3d(
            x=[c0[0], tip[0]],
            y=[c0[1], tip[1]],
            z=[c0[2], tip[2]],
            mode="lines+markers",
            line=dict(color="#1f77b4", width=6),
            marker=dict(size=[3, 6], color="#1f77b4"),
            name="plane normal",
        ),
        row=1,
        col=1,
    )

    for t in tags:
        r = row_by_id[t.tag_id]
        c = t.corners
        closed = np.vstack([c, c[0]])
        color_val = r["trans_err_cm"] / max_t_err_cm

        fig.add_trace(
            go.Scatter3d(
                x=closed[:, 0],
                y=closed[:, 1],
                z=closed[:, 2],
                mode="lines",
                line=dict(color="rgb(40,40,40)", width=3),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Mesh3d(
                x=c[:, 0],
                y=c[:, 1],
                z=c[:, 2],
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                intensity=[color_val] * 4,
                colorscale=colorscale,
                cmin=0.0,
                cmax=1.0,
                opacity=0.6,
                showscale=False,
                name=f"id {t.tag_id}",
                hovertemplate=(
                    f"id={t.tag_id}<br>"
                    f"n_err={r['normal_err_deg']:.3f}°<br>"
                    f"t_err={r['trans_err_cm']:.3f} cm"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        n_tag = np.array([r["nx"], r["ny"], r["nz"]], dtype=np.float64)
        tip_t = t.center + n_tag * (0.55 * arrow_len)
        fig.add_trace(
            go.Scatter3d(
                x=[t.center[0], tip_t[0]],
                y=[t.center[1], tip_t[1]],
                z=[t.center[2], tip_t[2]],
                mode="lines",
                line=dict(color="rgba(30,30,30,0.65)", width=3),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=[t.center[0]],
                y=[t.center[1]],
                z=[t.center[2]],
                mode="text",
                text=[str(t.tag_id)],
                textfont=dict(size=12, color="black"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            marker=dict(
                colorscale=colorscale,
                cmin=0.0,
                cmax=max_t_err_cm,
                color=[0],
                colorbar=dict(title="t_err [cm]", x=1.02, len=0.4, y=0.78),
                showscale=True,
                size=0.1,
            ),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    rows = result["rows"]
    ids = [str(r["tag_id"]) for r in rows]
    n_errs = [r["normal_err_deg"] for r in rows]
    t_errs_cm = [r["trans_err_cm"] for r in rows]

    fig.add_trace(
        go.Bar(
            x=ids,
            y=n_errs,
            name="normal_err [deg]",
            marker_color="#ff7f0e",
            opacity=0.9,
            text=[f"{v:.2f}" for v in n_errs],
            textposition="outside",
            hovertemplate="id=%{x}<br>n_err=%{y:.3f}°<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=ids,
            y=t_errs_cm,
            name="trans_err [cm]",
            marker_color="#2ca02c",
            opacity=0.9,
            text=[f"{v:.2f}" for v in t_errs_cm],
            textposition="outside",
            hovertemplate="id=%{x}<br>t_err=%{y:.3f} cm<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    # mean reference lines (Scatter, compatible with mixed scene/xy subplots)
    fig.add_trace(
        go.Scatter(
            x=[ids[0], ids[-1]],
            y=[result["mean_normal_err_deg"]] * 2,
            mode="lines",
            line=dict(color="#d62728", width=2, dash="dash"),
            name=f"mean n={result['mean_normal_err_deg']:.2f}°",
            hoverinfo="name",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[ids[0], ids[-1]],
            y=[result["mean_trans_err_cm"]] * 2,
            mode="lines",
            line=dict(color="#d62728", width=2, dash="dash"),
            name=f"mean t={result['mean_trans_err_cm']:.2f} cm",
            hoverinfo="name",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title=(
            "TagMap plane evaluation<br>"
            f"<sup>mean n_err={result['mean_normal_err_deg']:.3f}°, "
            f"mean t_err={result['mean_trans_err_cm']:.3f} cm | "
            f"n=[{n_plane[0]:.3f},{n_plane[1]:.3f},{n_plane[2]:.3f}] "
            f"d={d:.3f} | tags={result['num_tags']}</sup>"
        ),
        scene=dict(
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            zaxis_title="Z [m]",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=-1.6, z=1.0)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
        margin=dict(l=40, r=40, t=90, b=40),
        width=1400,
        height=1000,
        showlegend=True,
        bargap=0.25,
    )
    fig.update_xaxes(title_text="tag_id", row=2, col=1, type="category")
    fig.update_xaxes(title_text="tag_id", row=2, col=2, type="category")
    fig.update_yaxes(title_text="normal_err [deg]", row=2, col=1)
    fig.update_yaxes(title_text="trans_err [cm]", row=2, col=2)
    return fig


def save_html(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(path),
        include_plotlyjs="cdn",
        full_html=True,
        config=dict(displayModeBar=True, scrollZoom=True),
    )
    print(f"saved interactive 3D -> {path}")


def save_png(fig: go.Figure, path: Path) -> None:
    try:
        import plotly.io as pio

        path.parent.mkdir(parents=True, exist_ok=True)
        pio.write_image(fig, str(path), width=1400, height=780, scale=2)
        print(f"saved PNG snapshot -> {path}")
    except Exception as exc:
        print(f"PNG export skipped ({exc}); install kaleido: pip install kaleido")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate TagMap vs fitted plane GT (normal / translation) + viz."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"tag_map_corners.csv path (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="per-tag report CSV (default: <csv_dir>/tag_map_plane_eval.csv)",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        help="interactive HTML (default: <csv_dir>/tag_map_plane_eval.html)",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=None,
        help="optional PNG snapshot",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="skip HTML visualization",
    )
    args = parser.parse_args()

    out_dir = args.csv.resolve().parent
    out_csv = args.out or (out_dir / "tag_map_plane_eval.csv")
    out_html = args.html or (out_dir / "tag_map_plane_eval.html")

    tags = load_tags(args.csv)
    result = evaluate(tags)
    n = result["plane_n"]

    print(f"csv: {args.csv}")
    print(f"num_tags_with_corners: {result['num_tags']}")
    print(
        f"plane_gt: n=[{n[0]:.6f}, {n[1]:.6f}, {n[2]:.6f}]  d={result['plane_d']:.6f}"
    )
    print(
        f"normal_err_deg:  mean={result['mean_normal_err_deg']:.4f}  "
        f"median={result['median_normal_err_deg']:.4f}  "
        f"std={result['std_normal_err_deg']:.4f}  "
        f"max={result['max_normal_err_deg']:.4f}"
    )
    print(
        f"trans_err_cm:    mean={result['mean_trans_err_cm']:.4f}  "
        f"median={result['median_trans_err_cm']:.4f}  "
        f"std={result['std_trans_err_cm']:.4f}  "
        f"max={result['max_trans_err_cm']:.4f}"
    )
    print("per_tag:")
    for row in result["rows"]:
        print(
            f"  id={row['tag_id']:3d}  fixed={row['is_fixed']}  "
            f"n_err={row['normal_err_deg']:7.3f} deg  "
            f"t_err={row['trans_err_cm']:7.3f} cm"
        )

    write_report(out_csv, result)
    print(f"wrote: {out_csv}")

    if not args.no_html:
        fig = build_figure(tags, result)
        save_html(fig, out_html)
        if args.png is not None:
            save_png(fig, args.png)


if __name__ == "__main__":
    main()
