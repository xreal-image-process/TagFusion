#!/usr/bin/env python3
"""Visualize TagMap + ORB maps exported by TagMapExporter (interactive 3D).

Expected files under export_dir (default: Examples/TagFusion/XREAL_Glass/tag_export):
  - tag_map_corners.csv
  - orb_init_map_points.csv      (init snapshot; --final uses orb_final_map_points.csv)
  - tag_init_keyframes_tum.txt   (optional; --final: tag_final_keyframes_tum.txt)
  - orb_init_keyframes_tum.txt   (optional; --final: orb_final_keyframes_tum.txt)

Outputs (by default in export_dir):
  - init_maps_3d.html    interactive 3D (Plotly); --final -> final_maps_3d.html
  - init_maps_3d.ply     combined point cloud (optional); --final -> final_maps_3d.ply
  - init_maps_vis.png    static snapshot (optional, --png)

Usage:
  python3 scripts/visualize_init_maps.py
  python3 scripts/visualize_init_maps.py --export-dir path/to/tag_export
  python3 scripts/visualize_init_maps.py --final
  python3 scripts/visualize_init_maps.py --html out.html --ply out.ply --png out.png
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


SCRIPT_DIR = Path(__file__).resolve().parent
ORB_ROOT = SCRIPT_DIR.parent
DEFAULT_EXPORT_DIR = ORB_ROOT / "Examples/TagFusion/XREAL_Glass/tag_export"

TAG_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


@dataclass
class TagRecord:
    tag_id: int
    is_fixed: bool
    corners: np.ndarray  # (4, 3) or empty
    n_obs: int = 0
    mean_w_bar: float | None = None
    min_w_bar: float | None = None
    max_w_bar: float | None = None
    mean_alpha_w: float | None = None
    mean_w_s: float | None = None
    mean_w_theta: float | None = None
    mean_w_amb: float | None = None
    mean_w_obs: float | None = None
    mean_area: float | None = None
    weight_approx: bool = False


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


def weight_rgb(w: float | None) -> tuple[int, int, int]:
    """Red (suppressed) -> yellow -> green (full weight). Gray if missing."""
    if w is None:
        return (127, 127, 127)
    w = max(0.0, min(1.0, float(w)))
    if w < 0.5:
        t = w * 2.0
        return (242, int(51 + t * (191 - 51)), 26)
    t = (w - 0.5) * 2.0
    return (int(242 - t * (242 - 38)), int(191 + t * (139 - 191)), int(26 + t * (38 - 26)))


def weight_color(w: float | None) -> str:
    r, g, b = weight_rgb(w)
    return f"rgb({r},{g},{b})"


@dataclass
class InitMapData:
    tags: list[TagRecord]
    orb_pts: np.ndarray
    tag_traj: np.ndarray
    orb_traj: np.ndarray


def load_tag_corners(path: Path) -> list[TagRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"missing tag map: {path}")

    tags: list[TagRecord] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag_id = int(row["tag_id"])
            is_fixed = int(row.get("is_fixed", "0")) != 0
            has_corners = int(row.get("has_corners", "0")) != 0
            if not has_corners:
                tags.append(
                    TagRecord(
                        tag_id,
                        is_fixed,
                        np.empty((0, 3)),
                        n_obs=_opt_int(row, "n_obs"),
                        mean_w_bar=_opt_float(row, "mean_w_bar"),
                        min_w_bar=_opt_float(row, "min_w_bar"),
                        max_w_bar=_opt_float(row, "max_w_bar"),
                        mean_alpha_w=_opt_float(row, "mean_alpha_w"),
                        mean_w_s=_opt_float(row, "mean_w_s"),
                        mean_w_theta=_opt_float(row, "mean_w_theta"),
                        mean_w_amb=_opt_float(row, "mean_w_amb"),
                        mean_w_obs=_opt_float(row, "mean_w_obs"),
                        mean_area=_opt_float(row, "mean_area"),
                    )
                )
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
            tags.append(
                TagRecord(
                    tag_id,
                    is_fixed,
                    corners,
                    n_obs=_opt_int(row, "n_obs"),
                    mean_w_bar=_opt_float(row, "mean_w_bar"),
                    min_w_bar=_opt_float(row, "min_w_bar"),
                    max_w_bar=_opt_float(row, "max_w_bar"),
                    mean_alpha_w=_opt_float(row, "mean_alpha_w"),
                    mean_w_s=_opt_float(row, "mean_w_s"),
                    mean_w_theta=_opt_float(row, "mean_w_theta"),
                    mean_w_amb=_opt_float(row, "mean_w_amb"),
                    mean_w_obs=_opt_float(row, "mean_w_obs"),
                    mean_area=_opt_float(row, "mean_area"),
                )
            )
    return tags


def _approx_w_bar(
    area: float,
    theta_deg: float,
    s_sat: float = 400.0,
    lambda_s: float = 3.0,
    theta0_deg: float = 60.0,
    lambda_theta: float = 1.0,
    w_min: float = 1e-3,
) -> tuple[float, float, float]:
    """Same w_S / w_θ as EvaluateTagFactorWeight; w_amb=1 (first_reg has no IPPE pair)."""
    xs = max(0.0, area) / max(s_sat, 1e-6)
    w_s = min(1.0, max(0.0, 1.0 - math.exp(-lambda_s * xs * xs)))
    xt = abs(theta_deg) / max(theta0_deg, 1e-6)
    w_th = min(1.0, max(0.0, math.exp(-lambda_theta * xt * xt)))
    w_bar = max(w_s * w_th, w_min)
    return w_bar, w_s, w_th


def load_first_registration_approx(path: Path) -> dict[int, dict[str, float]]:
    """Per-tag mean approximate weight from tag_first_registration.csv."""
    if not path.is_file():
        return {}
    acc: dict[int, list[tuple[float, float, float, float]]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            tid = _opt_int(row, "tag_id", -1)
            area = _opt_float(row, "observed_area")
            theta = _opt_float(row, "axis_normal_angle_deg")
            if tid < 0 or area is None or theta is None:
                continue
            w_bar, w_s, w_th = _approx_w_bar(area, theta)
            acc.setdefault(tid, []).append((w_bar, w_s, w_th, area))
    out: dict[int, dict[str, float]] = {}
    for tid, rows in acc.items():
        n = float(len(rows))
        out[tid] = {
            "n_obs": float(len(rows)),
            "mean_w_bar": sum(r[0] for r in rows) / n,
            "min_w_bar": min(r[0] for r in rows),
            "mean_w_s": sum(r[1] for r in rows) / n,
            "mean_w_theta": sum(r[2] for r in rows) / n,
            "mean_area": sum(r[3] for r in rows) / n,
        }
    return out


def fill_missing_tag_weights(
    tags: list[TagRecord], first_reg: dict[int, dict[str, float]]
) -> list[TagRecord]:
    filled: list[TagRecord] = []
    for tag in tags:
        if tag.mean_w_bar is not None or tag.tag_id not in first_reg:
            filled.append(tag)
            continue
        approx = first_reg[tag.tag_id]
        filled.append(
            replace(
                tag,
                n_obs=max(tag.n_obs, int(approx["n_obs"])),
                mean_w_bar=approx["mean_w_bar"],
                min_w_bar=approx["min_w_bar"],
                mean_w_s=approx["mean_w_s"],
                mean_w_theta=approx["mean_w_theta"],
                mean_area=approx["mean_area"],
                mean_w_obs=approx["mean_w_bar"],
                mean_alpha_w=3.0 * approx["mean_w_bar"],
                weight_approx=True,
            )
        )
    return filled


def filter_orb_outliers(
    orb_pts: np.ndarray,
    tags: list[TagRecord],
    trajs: list[np.ndarray],
    mad_k: float = 6.0,
) -> tuple[np.ndarray, int, float]:
    """Drop far ORB init points that blow up the 3D view.

    Distance is measured from the Tag/KF median when available, else ORB median.
    Keep if dist <= median + mad_k * 1.4826 * MAD, capped by 3x Tag/KF span.
    """
    if orb_pts.size == 0 or orb_pts.shape[0] < 8:
        return orb_pts, 0, float("inf")

    ref_chunks: list[np.ndarray] = []
    for tag in tags:
        if tag.corners.size:
            ref_chunks.append(tag.corners)
    for traj in trajs:
        if traj.size:
            ref_chunks.append(traj)

    if ref_chunks:
        ref = np.vstack(ref_chunks)
        center = np.median(ref, axis=0)
        span = 0.5 * float(np.max(ref.max(axis=0) - ref.min(axis=0)))
    else:
        center = np.median(orb_pts, axis=0)
        span = 0.0

    dist = np.linalg.norm(orb_pts - center, axis=1)
    med = float(np.median(dist))
    mad = float(np.median(np.abs(dist - med)))
    sigma = 1.4826 * mad if mad > 1e-9 else 0.0
    thresh = med + mad_k * sigma if sigma > 0.0 else float("inf")
    if span > 1e-6:
        thresh = min(thresh, max(3.0 * span, med + 3.0 * sigma))

    keep = dist <= thresh
    n_drop = int((~keep).sum())
    return orb_pts[keep], n_drop, thresh


def load_orb_points(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing ORB map points: {path}")

    pts: list[list[float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pts.append([float(row["x"]), float(row["y"]), float(row["z"])])
    if not pts:
        return np.empty((0, 3))
    return np.asarray(pts, dtype=np.float64)


def load_tum_trajectory(path: Path) -> np.ndarray:
    """Return Nx3 camera centers (TUM: timestamp tx ty tz qx qy qz qw)."""
    if not path.is_file():
        return np.empty((0, 3))

    centers: list[list[float]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            centers.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not centers:
        return np.empty((0, 3))
    return np.asarray(centers, dtype=np.float64)


def load_init_maps(
    export_dir: Path,
    filter_orb: bool = True,
    orb_outlier_k: float = 6.0,
    fill_missing_weights: bool = True,
    final: bool = False,
) -> InitMapData:
    tags = load_tag_corners(export_dir / "tag_map_corners.csv")
    if fill_missing_weights:
        tags = fill_missing_tag_weights(
            tags, load_first_registration_approx(export_dir / "tag_first_registration.csv")
        )
    orb_name = "orb_final_map_points.csv" if final else "orb_init_map_points.csv"
    tag_kf_name = "tag_final_keyframes_tum.txt" if final else "tag_init_keyframes_tum.txt"
    orb_kf_name = "orb_final_keyframes_tum.txt" if final else "orb_init_keyframes_tum.txt"
    orb_pts = load_orb_points(export_dir / orb_name)
    tag_traj = load_tum_trajectory(export_dir / tag_kf_name)
    orb_traj = load_tum_trajectory(export_dir / orb_kf_name)
    if filter_orb:
        n_before = int(orb_pts.shape[0]) if orb_pts.size else 0
        orb_pts, n_drop, thresh = filter_orb_outliers(
            orb_pts, tags, [tag_traj, orb_traj], mad_k=orb_outlier_k
        )
        kind = "final" if final else "init"
        if n_drop:
            print(
                f"filtered {n_drop}/{n_before} far ORB {kind} points "
                f"(dist > {thresh:.2f} m from Tag/KF median)"
            )
    n_approx = sum(1 for t in tags if t.weight_approx)
    if n_approx:
        print(
            f"filled {n_approx} missing Tag weights from first_registration "
            "(KF observations were gone at export; label uses w≈)"
        )
    return InitMapData(tags=tags, orb_pts=orb_pts, tag_traj=tag_traj, orb_traj=orb_traj)


def combined_points(data: InitMapData) -> np.ndarray:
    chunks: list[np.ndarray] = []
    if data.orb_pts.size:
        chunks.append(data.orb_pts)
    tag_corners = [t.corners for t in data.tags if t.corners.size]
    if tag_corners:
        chunks.append(np.vstack(tag_corners))
    if data.tag_traj.size:
        chunks.append(data.tag_traj)
    if data.orb_traj.size:
        chunks.append(data.orb_traj)
    if not chunks:
        return np.empty((0, 3))
    return np.vstack(chunks)


def axis_ranges(points: np.ndarray, margin: float = 0.08) -> dict[str, list[float]]:
    if points.size == 0:
        return {"x": [-1, 1], "y": [-1, 1], "z": [-1, 1]}
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    if radius <= 0.0:
        radius = 0.1
    radius *= 1.0 + margin
    return {
        "x": [center[0] - radius, center[0] + radius],
        "y": [center[1] - radius, center[1] + radius],
        "z": [center[2] - radius, center[2] + radius],
    }


def tag_color(tag_id: int) -> str:
    return TAG_COLORS[tag_id % len(TAG_COLORS)]


def _tag_hover(tag: TagRecord) -> str:
    w_txt = "w=n/a"
    if tag.mean_w_bar is not None:
        w_txt = (
            f"w≈{tag.mean_w_bar:.3f}" if tag.weight_approx else f"w={tag.mean_w_bar:.3f}"
        )
    lines = [f"Tag {tag.tag_id}", w_txt, f"n_obs={tag.n_obs}"]
    if tag.weight_approx:
        lines.append("weight from first_registration (KF obs culled)")
    elif tag.n_obs == 0 and tag.mean_w_bar is None:
        lines.append("no remaining KF observations to score")
    if tag.min_w_bar is not None:
        lines.append(f"min_w={tag.min_w_bar:.3f}")
    if tag.mean_w_s is not None:
        lines.append(f"mean_wS={tag.mean_w_s:.3f}")
    if tag.mean_w_theta is not None:
        lines.append(f"mean_wTh={tag.mean_w_theta:.3f}")
    if tag.mean_w_amb is not None:
        lines.append(f"mean_wAmb={tag.mean_w_amb:.3f}")
    if tag.mean_alpha_w is not None:
        lines.append(f"mean_alpha_w={tag.mean_alpha_w:.3f}")
    return "<br>".join(lines)


def _quad_mesh_arrays(tags: list[TagRecord]):
    mx: list[float] = []
    my: list[float] = []
    mz: list[float] = []
    mi: list[int] = []
    mj: list[int] = []
    mk: list[int] = []
    facecolor: list[str] = []
    cx: list[float] = []
    cy: list[float] = []
    cz: list[float] = []
    labels: list[str] = []
    hovers: list[str] = []
    w_vals: list[float] = []
    for tag in tags:
        c = tag.corners
        color = (
            weight_color(tag.mean_w_bar)
            if tag.mean_w_bar is not None
            else tag_color(tag.tag_id)
        )
        base = len(mx)
        for pt in c:
            mx.append(float(pt[0]))
            my.append(float(pt[1]))
            mz.append(float(pt[2]))
        mi.extend([base, base])
        mj.extend([base + 1, base + 2])
        mk.extend([base + 2, base + 3])
        facecolor.extend([color, color])
        center = c.mean(axis=0)
        cx.append(float(center[0]))
        cy.append(float(center[1]))
        cz.append(float(center[2]))
        if tag.mean_w_bar is not None:
            labels.append(
                f"{tag.tag_id}  w≈{tag.mean_w_bar:.2f}"
                if tag.weight_approx
                else f"{tag.tag_id}  w={tag.mean_w_bar:.2f}"
            )
        else:
            labels.append(str(tag.tag_id))
        hovers.append(_tag_hover(tag))
        w_vals.append(float(tag.mean_w_bar) if tag.mean_w_bar is not None else 0.5)
    return mx, my, mz, mi, mj, mk, facecolor, cx, cy, cz, labels, hovers, w_vals


def build_plotly_figure(
    data: InitMapData, export_dir: Path, final: bool = False
) -> go.Figure:
    fig = go.Figure()
    combined = combined_points(data)

    if data.orb_pts.size:
        fig.add_trace(
            go.Scatter3d(
                x=data.orb_pts[:, 0],
                y=data.orb_pts[:, 1],
                z=data.orb_pts[:, 2],
                mode="markers",
                name=f"ORB map points ({data.orb_pts.shape[0]})",
                marker=dict(size=3, color="#4C72B0", opacity=0.65),
                hovertemplate="ORB<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            )
        )

    sel_tags = [t for t in data.tags if t.corners.shape == (4, 3)]

    if sel_tags:
        mx, my, mz, mi, mj, mk, facecolor, cx, cy, cz, labels, hovers, w_vals = (
            _quad_mesh_arrays(sel_tags)
        )
        fig.add_trace(
            go.Mesh3d(
                x=mx,
                y=my,
                z=mz,
                i=mi,
                j=mj,
                k=mk,
                facecolor=facecolor,
                opacity=0.50,
                showscale=False,
                hoverinfo="skip",
                name=f"Tag planes ({len(sel_tags)})",
            )
        )
        if not final:
            fig.add_trace(
                go.Scatter3d(
                    x=cx,
                    y=cy,
                    z=cz,
                    mode="markers+text",
                    name="selected Tag id",
                    text=labels,
                    textfont=dict(size=12, color="black"),
                    textposition="top center",
                    marker=dict(
                        size=4,
                        color=w_vals,
                        colorscale="RdYlGn",
                        cmin=0.0,
                        cmax=1.0,
                        colorbar=dict(title="mean w_bar", len=0.45, y=0.35),
                        showscale=any(t.mean_w_bar is not None for t in sel_tags),
                    ),
                    hovertext=hovers,
                    hovertemplate="%{hovertext}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
                    showlegend=False,
                )
            )

    if data.tag_traj.size and not final:
        fig.add_trace(
            go.Scatter3d(
                x=data.tag_traj[:, 0],
                y=data.tag_traj[:, 1],
                z=data.tag_traj[:, 2],
                mode="lines",
                name=(
                    f"Tag camera traj ({data.tag_traj.shape[0]})"
                    if final
                    else f"Tag init camera traj ({data.tag_traj.shape[0]})"
                ),
                line=dict(color="#2CA02C", width=6),
                hovertemplate="Tag camera<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            )
        )

    if data.orb_traj.size:
        fig.add_trace(
            go.Scatter3d(
                x=data.orb_traj[:, 0],
                y=data.orb_traj[:, 1],
                z=data.orb_traj[:, 2],
                mode="lines",
                name=(
                    f"Camera traj ({data.orb_traj.shape[0]})"
                    if final
                    else f"ORB init camera traj ({data.orb_traj.shape[0]})"
                ),
                line=dict(color="#D62728", width=4),
                hovertemplate="Camera<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            )
        )

    n_weighted = sum(1 for t in data.tags if t.mean_w_bar is not None)
    ranges = axis_ranges(combined)
    heading = (
        "Final map: Tag planes + ORB + camera traj"
        if final
        else "Init map: TagMap + ORB (same world frame)"
    )
    fig.update_layout(
        title=(
            f"{heading}<br>"
            f"<sup>{export_dir}<br>"
            f"tags={len(data.tags)} (with corners={len(sel_tags)}, "
            f"with weights={n_weighted}), "
            f"orb_points={data.orb_pts.shape[0]}"
            + (
                ""
                if final
                else "<br>selected fill = mean w_bar (red=suppressed, green=full)"
            )
            + "</sup>"
        ),
        scene=dict(
            xaxis=dict(title="X [m]", range=ranges["x"]),
            yaxis=dict(title="Y [m]", range=ranges["y"]),
            zaxis=dict(title="Z [m]", range=ranges["z"]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.6, y=-1.8, z=1.0)),
        ),
        legend=dict(
            x=0.01,
            y=0.98,
            bgcolor="rgba(255,255,255,0.75)",
            itemsizing="constant",
        ),
        margin=dict(l=0, r=0, t=90, b=0),
        width=1100,
        height=800,
    )
    return fig


def save_html(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(path),
        include_plotlyjs=True,
        full_html=True,
        config=dict(displayModeBar=True, scrollZoom=True),
    )
    print(f"saved interactive 3D -> {path}")


def save_png(fig: go.Figure, path: Path) -> None:
    try:
        import plotly.io as pio

        path.parent.mkdir(parents=True, exist_ok=True)
        pio.write_image(fig, str(path), width=1100, height=800, scale=2)
        print(f"saved PNG snapshot -> {path}")
    except Exception as exc:
        print(f"PNG export skipped ({exc}); install kaleido: pip install kaleido")


def save_ply(data: InitMapData, path: Path) -> None:
    """Write ASCII PLY: ORB points + Tag planes + camera trajectories as edges."""
    path.parent.mkdir(parents=True, exist_ok=True)

    vertices: list[tuple[float, float, float, int, int, int]] = []
    faces: list[tuple[int, int, int, int]] = []
    edges: list[tuple[int, int, int, int, int]] = []

    def append_quad(corners: np.ndarray, rgb: tuple[int, int, int]) -> None:
        base = len(vertices)
        r, g, b = rgb
        for c in corners:
            vertices.append((float(c[0]), float(c[1]), float(c[2]), r, g, b))
        faces.append((base, base + 1, base + 2, base + 3))

    def append_traj(pts: np.ndarray, rgb: tuple[int, int, int]) -> None:
        if pts.size == 0:
            return
        r, g, b = rgb
        base = len(vertices)
        for p in pts:
            vertices.append((float(p[0]), float(p[1]), float(p[2]), r, g, b))
        for i in range(base, len(vertices) - 1):
            edges.append((i, i + 1, r, g, b))

    for p in data.orb_pts:
        vertices.append((p[0], p[1], p[2], 76, 114, 176))

    for tag in data.tags:
        if tag.corners.shape == (4, 3):
            rgb = (
                weight_rgb(tag.mean_w_bar)
                if tag.mean_w_bar is not None
                else _hex_to_rgb(tag_color(tag.tag_id))
            )
            append_quad(tag.corners, rgb)

    append_traj(data.orb_traj, (214, 39, 40))
    append_traj(data.tag_traj, (44, 160, 44))

    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for x, y, z, r, g, b in vertices:
            f.write(f"{x:.9f} {y:.9f} {z:.9f} {r} {g} {b}\n")
        for i0, i1, i2, i3 in faces:
            f.write(f"4 {i0} {i1} {i2} {i3}\n")
        for v0, v1, r, g, b in edges:
            f.write(f"{v0} {v1} {r} {g} {b}\n")

    print(
        f"saved PLY -> {path} (vertices={len(vertices)}, "
        f"tag_planes={len(faces)}, traj_edges={len(edges)})"
    )


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def visualize_3d(
    export_dir: Path,
    html_path: Path,
    png_path: Path | None,
    ply_path: Path | None,
    open_browser: bool,
    filter_orb: bool = True,
    orb_outlier_k: float = 6.0,
    final: bool = False,
) -> None:
    data = load_init_maps(
        export_dir,
        filter_orb=filter_orb,
        orb_outlier_k=orb_outlier_k,
        fill_missing_weights=not final,
        final=final,
    )
    fig = build_plotly_figure(data, export_dir, final=final)

    save_html(fig, html_path)
    if png_path is not None:
        save_png(fig, png_path)
    if ply_path is not None:
        save_ply(data, ply_path)

    if open_browser:
        import webbrowser

        webbrowser.open(html_path.as_uri())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate interactive 3D visualization for TagMap + ORB maps."
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="visualize shutdown final map (orb_final_*.csv / *_final_keyframes_tum.txt)",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=DEFAULT_EXPORT_DIR,
        help=f"input export directory (default: {DEFAULT_EXPORT_DIR})",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        help="output HTML path (default: <export-dir>/init_maps_3d.html or final_maps_3d.html)",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=None,
        help="optional static PNG snapshot",
    )
    parser.add_argument(
        "--ply",
        type=Path,
        default=None,
        help="optional PLY point cloud export",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="open HTML in browser after generation",
    )
    parser.add_argument(
        "--no-ply",
        action="store_true",
        help="skip default PLY export",
    )
    parser.add_argument(
        "--no-orb-cull",
        action="store_true",
        help="keep far ORB init outliers (default: cull by MAD vs Tag/KF median)",
    )
    parser.add_argument(
        "--orb-outlier-k",
        type=float,
        default=6.0,
        help="MAD multiplier for ORB outlier distance (default: 6)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export_dir = args.export_dir.resolve()
    if not export_dir.is_dir():
        print(f"export dir not found: {export_dir}")
        return 1

    stem = "final_maps_3d" if args.final else "init_maps_3d"
    html_path = (args.html or (export_dir / f"{stem}.html")).resolve()
    png_path = args.png.resolve() if args.png else None
    if args.no_ply or (args.final and args.ply is None):
        ply_path = None
    elif args.ply is not None:
        ply_path = args.ply.resolve()
    else:
        ply_path = (export_dir / f"{stem}.ply").resolve()

    try:
        visualize_3d(
            export_dir=export_dir,
            html_path=html_path,
            png_path=png_path,
            ply_path=ply_path,
            open_browser=args.open,
            filter_orb=not args.no_orb_cull,
            orb_outlier_k=args.orb_outlier_k,
            final=args.final,
        )
    except FileNotFoundError as exc:
        print(exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
