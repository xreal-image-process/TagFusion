#!/usr/bin/env python3
"""Evaluate TagMap coplanarity on calibration-board planes (selected vs mirror).

Reads tag_map_corners.csv and tag_map_mirror_corners.csv from TagMapExporter.

Default planes (XREAL glass calibr board):
  P1: 30, 91, 93, 23
  P2: 3, 2, 6, 5, 25, 4, 92
  P3: 21, 20, 19, 18, 17, 16, 35

Pipeline (per plane, per solution):
  1) Per-tag center / normal from corners.
  2) Fit plane to tag centers (SVD).
  3) Report normal error (deg) and translation error (cm).
  4) Interactive HTML: selected/mirror 3D + grouped bar charts.

Usage:
  python3 scripts/evaluate_tag_map_calibr_planes.py
  python3 scripts/evaluate_tag_map_calibr_planes.py --csv path/to/tag_map_corners.csv
  python3 scripts/evaluate_tag_map_calibr_planes.py --csv ... --mirror path/to/tag_map_mirror_corners.csv
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

CALIBR_PLANES: list[tuple[str, tuple[int, ...]]] = [
    ("P1", (30, 91, 93, 23)),
    ("P2", (3, 2, 6, 5, 25, 4, 92)),
    ("P3", (21, 20, 19, 18, 17, 16, 35)),
]

PLANE_COLORS = {
    "P1": "#ff7f0e",
    "P2": "#2ca02c",
    "P3": "#9467bd",
}


@dataclass
class TagGeom:
    tag_id: int
    is_fixed: bool
    corners: np.ndarray  # (4, 3)
    center: np.ndarray  # (3,)
    normal: np.ndarray  # (3,) unit
    n_obs: int = 0
    mean_w_bar: float | None = None
    min_w_bar: float | None = None
    mean_alpha_w: float | None = None
    mean_w_s: float | None = None
    mean_w_theta: float | None = None
    mean_w_amb: float | None = None


def _opt_float(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key)
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return float(text)


def _opt_int(row: dict[str, str], key: str, default: int = 0) -> int:
    raw = row.get(key)
    if raw is None:
        return default
    text = str(raw).strip()
    if not text:
        return default
    return int(float(text))


def weight_color(w: float | None) -> str:
    if w is None:
        return "rgb(127,127,127)"
    w = max(0.0, min(1.0, float(w)))
    if w < 0.5:
        t = w * 2.0
        return f"rgb(242,{int(51 + t * (191 - 51))},26)"
    t = (w - 0.5) * 2.0
    return f"rgb({int(242 - t * (242 - 38))},{int(191 + t * (139 - 191))},{int(26 + t * (38 - 26))})"


def load_tags(path: Path) -> dict[int, TagGeom]:
    if not path.is_file():
        raise FileNotFoundError(f"missing tag map: {path}")

    tags: dict[int, TagGeom] = {}
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
            tag_id = int(row["tag_id"])
            tags[tag_id] = TagGeom(
                tag_id=tag_id,
                is_fixed=int(row.get("is_fixed", "0")) != 0,
                corners=corners,
                center=center,
                normal=n,
                n_obs=_opt_int(row, "n_obs"),
                mean_w_bar=_opt_float(row, "mean_w_bar"),
                min_w_bar=_opt_float(row, "min_w_bar"),
                mean_alpha_w=_opt_float(row, "mean_alpha_w"),
                mean_w_s=_opt_float(row, "mean_w_s"),
                mean_w_theta=_opt_float(row, "mean_w_theta"),
                mean_w_amb=_opt_float(row, "mean_w_amb"),
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
                "n_obs": t.n_obs,
                "mean_w_bar": t.mean_w_bar,
                "min_w_bar": t.min_w_bar,
                "mean_alpha_w": t.mean_alpha_w,
                "mean_w_s": t.mean_w_s,
                "mean_w_theta": t.mean_w_theta,
                "mean_w_amb": t.mean_w_amb,
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


def select_plane_tags(
    all_tags: dict[int, TagGeom], ids: tuple[int, ...]
) -> tuple[list[TagGeom], list[int]]:
    found = [all_tags[i] for i in ids if i in all_tags]
    missing = [i for i in ids if i not in all_tags]
    return found, missing


def evaluate_planes(
    all_tags: dict[int, TagGeom], solution: str
) -> tuple[list[dict], list[dict]]:
    plane_results: list[dict] = []
    all_rows: list[dict] = []
    for name, ids in CALIBR_PLANES:
        tags, missing = select_plane_tags(all_tags, ids)
        entry = {
            "plane": name,
            "ids": ids,
            "color": PLANE_COLORS[name],
            "tags": tags,
            "missing": missing,
            "result": None,
            "solution": solution,
        }
        if len(tags) < 3:
            print(
                f"  skip {solution}/{name}: need >=3 tags with corners, "
                f"got {len(tags)} (missing={missing})"
            )
            plane_results.append(entry)
            continue
        if missing:
            print(f"  warn {solution}/{name}: missing tags {missing}")
        result = evaluate(tags)
        entry["result"] = result
        plane_results.append(entry)
        for row in result["rows"]:
            rec = dict(row)
            rec["solution"] = solution
            rec["plane"] = name
            all_rows.append(rec)
    return plane_results, all_rows


def write_report(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "solution",
        "plane",
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
        "n_obs",
        "mean_w_bar",
        "min_w_bar",
        "mean_alpha_w",
        "mean_w_s",
        "mean_w_theta",
        "mean_w_amb",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    normal_path = path.with_name(path.stem + "_normal.csv")
    trans_path = path.with_name(path.stem + "_translation.csv")
    split_fields_n = ["solution", "plane", "tag_id", "is_fixed", "normal_err_deg"]
    split_fields_t = ["solution", "plane", "tag_id", "is_fixed", "trans_err_cm"]
    with normal_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=split_fields_n)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in split_fields_n})
    with trans_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=split_fields_t)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in split_fields_t})
    print(f"wrote: {normal_path}")
    print(f"wrote: {trans_path}")


def _plane_mesh(
    n: np.ndarray, d: float, centers: np.ndarray, scale: float = 1.15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    centroid = centers.mean(axis=0)
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


def _add_scene(
    fig: go.Figure,
    plane_results: list[dict],
    row: int,
    col: int,
    legend_group: str,
    show_legend: bool,
) -> None:
    all_centers = []
    for pr in plane_results:
        if pr["result"] is None or not pr["tags"]:
            continue
        all_centers.append(np.stack([t.center for t in pr["tags"]], axis=0))
    if not all_centers:
        return
    span = float(
        np.linalg.norm(
            np.vstack(all_centers).max(0) - np.vstack(all_centers).min(0)
        )
    )
    arrow_len = max(0.08, 0.18 * span)

    for pr in plane_results:
        result = pr["result"]
        tags: list[TagGeom] = pr["tags"]
        if result is None or not tags:
            continue
        n_plane = result["plane_n"]
        d = result["plane_d"]
        color = pr["color"]
        centers = np.stack([t.center for t in tags], axis=0)
        row_by_id = {r["tag_id"]: r for r in result["rows"]}

        px, py, pz = _plane_mesh(n_plane, d, centers)
        fig.add_trace(
            go.Mesh3d(
                x=px,
                y=py,
                z=pz,
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color=color,
                opacity=0.22,
                name=f"{pr['plane']} plane",
                legendgroup=f"{legend_group}-{pr['plane']}",
                showlegend=show_legend,
            ),
            row=row,
            col=col,
        )

        c0 = centers.mean(axis=0)
        c0 = c0 - (np.dot(n_plane, c0) + d) * n_plane
        tip = c0 + n_plane * arrow_len
        fig.add_trace(
            go.Scatter3d(
                x=[c0[0], tip[0]],
                y=[c0[1], tip[1]],
                z=[c0[2], tip[2]],
                mode="lines+markers",
                line=dict(color=color, width=6),
                marker=dict(size=[3, 6], color=color),
                name=f"{pr['plane']} n",
                legendgroup=f"{legend_group}-{pr['plane']}",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        for t in tags:
            r = row_by_id[t.tag_id]
            c = t.corners
            closed = np.vstack([c, c[0]])
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
                row=row,
                col=col,
            )
            fill = weight_color(t.mean_w_bar)
            w_txt = (
                f"w={t.mean_w_bar:.3f}" if t.mean_w_bar is not None else "w=n/a"
            )
            hover = (
                f"{pr['plane']} id={t.tag_id}<br>"
                f"{w_txt}<br>"
                f"n_obs={t.n_obs}<br>"
                f"n_err={r['normal_err_deg']:.3f}°<br>"
                f"t_err={r['trans_err_cm']:.3f} cm"
            )
            if t.mean_w_s is not None:
                hover += (
                    f"<br>mean_wS={t.mean_w_s:.3f}"
                    f"<br>mean_wTh={t.mean_w_theta:.3f}"
                    f"<br>mean_wAmb={t.mean_w_amb:.3f}"
                )
            hover += "<extra></extra>"
            fig.add_trace(
                go.Mesh3d(
                    x=c[:, 0],
                    y=c[:, 1],
                    z=c[:, 2],
                    i=[0, 0],
                    j=[1, 2],
                    k=[2, 3],
                    color=fill,
                    opacity=0.62,
                    name=f"{pr['plane']} id {t.tag_id}",
                    legendgroup=f"{legend_group}-{pr['plane']}",
                    showlegend=False,
                    hovertemplate=hover,
                ),
                row=row,
                col=col,
            )
            n_tag = np.array([r["nx"], r["ny"], r["nz"]], dtype=np.float64)
            tip_t = t.center + n_tag * (0.45 * arrow_len)
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
                row=row,
                col=col,
            )
            label = (
                f"{t.tag_id} w={t.mean_w_bar:.2f}"
                if t.mean_w_bar is not None
                else str(t.tag_id)
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[t.center[0]],
                    y=[t.center[1]],
                    z=[t.center[2]],
                    mode="text",
                    text=[label],
                    textfont=dict(size=11, color="black"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )


def _bar_xy(
    sel_planes: list[dict], mir_planes: list[dict], key: str
) -> tuple[list[str], list[float | None], list[float | None], list[str]]:
    labels: list[str] = []
    sel_vals: list[float | None] = []
    mir_vals: list[float | None] = []
    colors: list[str] = []
    mir_by_plane = {pr["plane"]: pr for pr in mir_planes}
    for pr in sel_planes:
        sel_rows = (
            {r["tag_id"]: r for r in pr["result"]["rows"]} if pr["result"] else {}
        )
        mir_pr = mir_by_plane.get(pr["plane"])
        mir_rows = (
            {r["tag_id"]: r for r in mir_pr["result"]["rows"]}
            if mir_pr and mir_pr["result"]
            else {}
        )
        for tid in pr["ids"]:
            labels.append(f"{pr['plane']}-{tid}")
            colors.append(pr["color"])
            sel_vals.append(sel_rows[tid][key] if tid in sel_rows else None)
            mir_vals.append(mir_rows[tid][key] if tid in mir_rows else None)
    return labels, sel_vals, mir_vals, colors


def _summary_line(plane_results: list[dict]) -> str:
    parts = []
    for pr in plane_results:
        r = pr["result"]
        if r is None:
            parts.append(f"{pr['plane']}: n/a")
            continue
        parts.append(
            f"{pr['plane']} n={r['mean_normal_err_deg']:.2f}° "
            f"t={r['mean_trans_err_cm']:.2f}cm"
        )
    return " | ".join(parts)


def build_figure(sel_planes: list[dict], mir_planes: list[dict]) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "scene"}, {"type": "scene"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        row_heights=[0.56, 0.44],
        subplot_titles=(
            "Selected TagMap + fitted planes",
            "Mirror TagMap + fitted planes",
            "Normal error [deg]",
            "Translation error [cm]",
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )

    _add_scene(fig, sel_planes, 1, 1, "sel", True)
    _add_scene(fig, mir_planes, 1, 2, "mir", False)

    labels, n_sel, n_mir, _ = _bar_xy(sel_planes, mir_planes, "normal_err_deg")
    _, t_sel, t_mir, _ = _bar_xy(sel_planes, mir_planes, "trans_err_cm")

    fig.add_trace(
        go.Bar(
            x=labels,
            y=n_sel,
            name="selected n_err",
            marker_color="#1f77b4",
            opacity=0.9,
            text=[f"{v:.2f}" if v is not None else "" for v in n_sel],
            textposition="outside",
            hovertemplate="id=%{x}<br>selected n_err=%{y:.3f}°<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=n_mir,
            name="mirror n_err",
            marker_color="#d62728",
            opacity=0.85,
            text=[f"{v:.2f}" if v is not None else "" for v in n_mir],
            textposition="outside",
            hovertemplate="id=%{x}<br>mirror n_err=%{y:.3f}°<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=t_sel,
            name="selected t_err",
            marker_color="#1f77b4",
            opacity=0.9,
            text=[f"{v:.2f}" if v is not None else "" for v in t_sel],
            textposition="outside",
            hovertemplate="id=%{x}<br>selected t_err=%{y:.3f} cm<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=t_mir,
            name="mirror t_err",
            marker_color="#d62728",
            opacity=0.85,
            text=[f"{v:.2f}" if v is not None else "" for v in t_mir],
            textposition="outside",
            hovertemplate="id=%{x}<br>mirror t_err=%{y:.3f} cm<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    scene = dict(
        xaxis_title="X [m]",
        yaxis_title="Y [m]",
        zaxis_title="Z [m]",
        aspectmode="data",
        camera=dict(eye=dict(x=1.5, y=-1.6, z=1.0)),
    )
    fig.update_layout(
        title=(
            "Calibr-board multi-plane eval (selected vs unused IPPE mirror)<br>"
            f"<sup>selected: {_summary_line(sel_planes)}<br>"
            f"mirror: {_summary_line(mir_planes)}<br>"
            "Tag fill/label = mean factor weight w_bar (red=suppressed, green=full)</sup>"
        ),
        scene=scene,
        scene2=scene,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
        margin=dict(l=40, r=40, t=110, b=40),
        width=1600,
        height=1100,
        showlegend=True,
        barmode="group",
        bargap=0.25,
    )
    fig.update_xaxes(title_text="plane-tag_id", row=2, col=1, type="category")
    fig.update_xaxes(title_text="plane-tag_id", row=2, col=2, type="category")
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
        pio.write_image(fig, str(path), width=1600, height=1100, scale=2)
        print(f"saved PNG snapshot -> {path}")
    except Exception as exc:
        print(f"PNG export skipped ({exc}); install kaleido: pip install kaleido")


def print_solution(label: str, plane_results: list[dict]) -> None:
    print(f"\n== {label} ==")
    for pr in plane_results:
        r = pr["result"]
        print(f"  {pr['plane']} ids={list(pr['ids'])} missing={pr['missing']}")
        if r is None:
            continue
        n = r["plane_n"]
        print(
            f"    plane: n=[{n[0]:.6f}, {n[1]:.6f}, {n[2]:.6f}]  d={r['plane_d']:.6f}"
        )
        print(
            f"    normal_err_deg: mean={r['mean_normal_err_deg']:.4f}  "
            f"median={r['median_normal_err_deg']:.4f}  "
            f"std={r['std_normal_err_deg']:.4f}  max={r['max_normal_err_deg']:.4f}"
        )
        print(
            f"    trans_err_cm:   mean={r['mean_trans_err_cm']:.4f}  "
            f"median={r['median_trans_err_cm']:.4f}  "
            f"std={r['std_trans_err_cm']:.4f}  max={r['max_trans_err_cm']:.4f}"
        )
        for row in r["rows"]:
            print(
                f"      id={row['tag_id']:3d}  "
                f"n_err={row['normal_err_deg']:7.3f} deg  "
                f"t_err={row['trans_err_cm']:7.3f} cm"
            )


def print_compare(sel_planes: list[dict], mir_planes: list[dict]) -> None:
    mir_by = {pr["plane"]: pr for pr in mir_planes}
    print("\n== selected vs mirror (lower is better) ==")
    for pr in sel_planes:
        sr = pr["result"]
        mr = mir_by.get(pr["plane"], {}).get("result") if mir_by.get(pr["plane"]) else None
        if sr is None and mr is None:
            print(f"  {pr['plane']}: both n/a")
            continue
        if sr is None:
            print(f"  {pr['plane']}: selected n/a | mirror n={mr['mean_normal_err_deg']:.3f}° "
                  f"t={mr['mean_trans_err_cm']:.3f}cm")
            continue
        if mr is None:
            print(f"  {pr['plane']}: selected n={sr['mean_normal_err_deg']:.3f}° "
                  f"t={sr['mean_trans_err_cm']:.3f}cm | mirror n/a")
            continue
        n_win = "mirror" if mr["mean_normal_err_deg"] < sr["mean_normal_err_deg"] else "selected"
        t_win = "mirror" if mr["mean_trans_err_cm"] < sr["mean_trans_err_cm"] else "selected"
        print(
            f"  {pr['plane']}: n  sel={sr['mean_normal_err_deg']:.3f}°  "
            f"mir={mr['mean_normal_err_deg']:.3f}°  better={n_win}"
        )
        print(
            f"         t  sel={sr['mean_trans_err_cm']:.3f}cm  "
            f"mir={mr['mean_trans_err_cm']:.3f}cm  better={t_win}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate calibr-board planes (selected vs unused IPPE mirror)."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"selected tag_map_corners.csv (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--mirror",
        type=Path,
        default=None,
        help="mirror CSV (default: <csv_dir>/tag_map_mirror_corners.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="per-tag report CSV (default: <csv_dir>/tag_map_calibr_planes_eval.csv)",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        help="interactive HTML (default: <csv_dir>/tag_map_calibr_planes_eval.html)",
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
    mirror_csv = args.mirror or (out_dir / "tag_map_mirror_corners.csv")
    out_csv = args.out or (out_dir / "tag_map_calibr_planes_eval.csv")
    out_html = args.html or (out_dir / "tag_map_calibr_planes_eval.html")

    print(f"selected csv: {args.csv}")
    print(f"mirror   csv: {mirror_csv}")
    sel_tags = load_tags(args.csv)
    mir_tags: dict[int, TagGeom] = {}
    if mirror_csv.is_file():
        mir_tags = load_tags(mirror_csv)
    else:
        print(f"warn: mirror csv missing, evaluate selected only: {mirror_csv}")

    sel_planes, sel_rows = evaluate_planes(sel_tags, "selected")
    mir_planes, mir_rows = evaluate_planes(mir_tags, "mirror") if mir_tags else ([], [])
    if not mir_planes:
        mir_planes = [
            {
                "plane": name,
                "ids": ids,
                "color": PLANE_COLORS[name],
                "tags": [],
                "missing": list(ids),
                "result": None,
                "solution": "mirror",
            }
            for name, ids in CALIBR_PLANES
        ]

    print_solution("selected", sel_planes)
    print_solution("mirror", mir_planes)
    print_compare(sel_planes, mir_planes)

    all_rows = sel_rows + mir_rows
    write_report(out_csv, all_rows)
    print(f"wrote: {out_csv}")

    if not args.no_html:
        fig = build_figure(sel_planes, mir_planes)
        save_html(fig, out_html)
        if args.png is not None:
            save_png(fig, args.png)


if __name__ == "__main__":
    main()
