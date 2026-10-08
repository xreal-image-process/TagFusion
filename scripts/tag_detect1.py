#!/usr/bin/env python3
"""Detect AprilTags via Thirdparty/apriltag (ethz_apriltag2).

Backend: scripts/ethz_tag_detect CLI — same as ua_tag::AprilTagDetector /
tag_detect.py. OpenCV ArUco path removed.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CLI = SCRIPT_DIR / "ethz_tag_detect"
DEFAULT_IMAGE_DIR = Path(
    "/home/geneta/dataset/20260227-room-patrol-11-1/raw_data/glass/cam0/images"
)
DEFAULT_GLASS_CONFIG = Path(
    "/home/geneta/dataset/20260227-room-patrol-11-1/raw_data/glass/glass_config.json"
)

SUPPORTED_FAMILIES = ("tag36h11", "tag36h9", "tag25h9", "tag25h7", "tag16h5")


class EthzAprilTagDetector:
    """ethz_apriltag2 detector via ethz_tag_detect CLI."""

    def __init__(
        self,
        family: str = "tag36h11",
        hamming: int = 2,
        black_border: int = 1,
        cli_path: Path = DEFAULT_CLI,
        **_ignored,
    ) -> None:
        if family not in SUPPORTED_FAMILIES:
            raise ValueError(
                f"unsupported family: {family}; choose from {SUPPORTED_FAMILIES}"
            )
        if not cli_path.is_file():
            raise FileNotFoundError(
                f"ethz_tag_detect not found: {cli_path}\n"
                "Build: cmake --build build --target ethz_tag_detect"
            )
        self.family = family
        self.hamming = int(hamming)
        self.black_border = int(black_border)
        self.cli_path = cli_path

    def detect(self, gray: np.ndarray, scale: float = 1.0) -> tuple[list[dict], int]:
        if gray.ndim != 2:
            raise ValueError("detect() expects a single-channel grayscale image")
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
        if scale <= 0:
            raise ValueError("scale must be positive")

        if scale > 1.0 + 1e-9:
            detect_image = cv2.resize(
                gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
        else:
            detect_image = gray

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cv2.imwrite(str(tmp_path), detect_image)
            cmd = [
                str(self.cli_path),
                "--family",
                self.family,
                "--hamming",
                str(self.hamming),
                "--black-border",
                str(self.black_border),
                str(tmp_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ethz_tag_detect failed: {(proc.stderr or '').strip()}"
                )

            out: list[dict] = []
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                corners = np.asarray(obj["corners"], dtype=np.float64) / scale
                center = corners.mean(axis=0)
                perimeter = float(obj.get("perimeter", 0.0)) / scale
                area = float(abs(cv2.contourArea(corners.astype(np.float32))))
                out.append(
                    {
                        "id": int(obj["id"]),
                        "center": (float(center[0]), float(center[1])),
                        "corners": corners,
                        "perimeter": perimeter,
                        "area": area,
                        "hamming": int(obj["hamming"]),
                    }
                )
            return out, 0
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass



def merge_detections(
    detections: list[dict],
    center_threshold: float = 4.0,
) -> list[dict]:
    """Merge repeated detections from multiple threshold passes.

    Detections whose centers are close are treated as the same physical marker.
    The ID with the most votes is selected; ties prefer the larger-area quad.
    """
    clusters: list[list[dict]] = []
    for det in detections:
        center = np.asarray(det["center"], dtype=np.float64)
        assigned = False
        for cluster in clusters:
            ref = np.asarray(cluster[0]["center"], dtype=np.float64)
            if np.linalg.norm(center - ref) <= center_threshold:
                cluster.append(det)
                assigned = True
                break
        if not assigned:
            clusters.append([det])

    merged: list[dict] = []
    for cluster in clusters:
        votes: dict[int, int] = {}
        for det in cluster:
            votes[det["id"]] = votes.get(det["id"], 0) + 1
        best_id = max(
            votes,
            key=lambda marker_id: (
                votes[marker_id],
                max(
                    d["area"] for d in cluster
                    if d["id"] == marker_id
                ),
            ),
        )
        candidates = [d for d in cluster if d["id"] == best_id]
        best = max(candidates, key=lambda d: d["area"]).copy()
        best["votes"] = votes[best_id]
        best["passes"] = len(cluster)
        merged.append(best)

    merged.sort(key=lambda d: (d["center"][1], d["center"][0]))
    return merged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect AprilTags in Glass cam0 images with OpenCV ArUcoDetector"
    )
    p.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    p.add_argument("--image", type=Path, default=None, help="Single image path")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1, help="Exclusive end index, -1=all")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument(
        "--family",
        default="tag36h11",
        choices=list(SUPPORTED_FAMILIES),
        help="ethz_apriltag2 family",
    )
    p.add_argument("--hamming", type=int, default=2)
    p.add_argument("--black-border", type=int, default=1)
    p.add_argument("--cli", type=Path, default=DEFAULT_CLI)

    # Successful low-resolution configuration.
    p.add_argument(
        "--upsample",
        type=float,
        default=4.0,
        help="Detection-only scale; corners are mapped back afterward",
    )
    p.add_argument("--adaptive-thresh-min", type=int, default=3)
    p.add_argument(
        "--adaptive-thresh-max",
        type=int,
        default=95,
        help="Max adaptive threshold window on the upsampled image",
    )
    p.add_argument("--adaptive-thresh-step", type=int, default=4)
    p.add_argument("--adaptive-thresh-constant", type=float, default=7.0)
    p.add_argument(
        "--adaptive-thresh-constants",
        type=str,
        default="5,7,9,11",
        help=(
            "Comma-separated threshold constants for multi-pass detection. "
            "Use an empty string to run only --adaptive-thresh-constant."
        ),
    )
    p.add_argument(
        "--merge-center-threshold",
        type=float,
        default=4.0,
        help="Center distance in original-image pixels for merging pass duplicates",
    )
    p.add_argument("--min-marker-perimeter-rate", type=float, default=0.01)
    p.add_argument("--max-marker-perimeter-rate", type=float, default=4.0)
    p.add_argument("--polygonal-approx-accuracy-rate", type=float, default=0.05)
    p.add_argument("--min-corner-distance-rate", type=float, default=0.03)
    p.add_argument("--min-distance-to-border", type=int, default=1)
    p.add_argument("--min-marker-distance-rate", type=float, default=0.05)
    p.add_argument(
        "--perspective-remove-pixel-per-cell",
        type=int,
        default=8,
        help="Canonical pixels sampled per marker cell",
    )
    p.add_argument(
        "--perspective-remove-ignored-margin-per-cell",
        type=float,
        default=0.20,
        help="Ignore cell-edge pixels during bit decoding",
    )
    p.add_argument(
        "--max-erroneous-bits-in-border-rate",
        type=float,
        default=0.50,
    )
    p.add_argument(
        "--min-otsu-std-dev",
        type=float,
        default=2.0,
        help="Lower value keeps Otsu decoding active on low-contrast tags",
    )
    p.add_argument("--error-correction-rate", type=float, default=0.8)
    # Kept for CLI compatibility (ignored; ethz backend does not use these)
    p.add_argument("--corner-refine", default="none")
    p.add_argument("--detect-inverted", action="store_true")

    p.add_argument(
        "--clahe",
        dest="use_clahe",
        action="store_true",
        default=False,
        help="Enable CLAHE (off by default to reproduce the successful setting)",
    )
    p.add_argument("--no-clahe", dest="use_clahe", action="store_false")
    p.add_argument("--clahe-clip", type=float, default=1.5)
    p.add_argument("--clahe-tile", type=int, default=8)

    p.add_argument(
        "--undistort",
        dest="undistort",
        action="store_true",
        default=False,
        help="Undistort with glass_config.json before detection (off by default)",
    )
    p.add_argument("--no-undistort", dest="undistort", action="store_false")
    p.add_argument("--glass-config", type=Path, default=DEFAULT_GLASS_CONFIG)
    p.add_argument("--cam-device", type=int, default=1)

    p.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR / "tag_test_out_opencv",
        help="Save vis/ and detections.csv",
    )
    p.add_argument("--show", action="store_true", help="q/ESC quit, space pause")
    p.add_argument(
        "--only-hits",
        dest="only_hits",
        action="store_true",
        default=True,
        help="Only save/show frames with detections (default)",
    )
    p.add_argument("--no-only-hits", dest="only_hits", action="store_false")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def list_images(image_dir: Path) -> list[Path]:
    exts = {".pgm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name)
    return files


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"unsupported channel count: {image.shape[2]}")


def apply_clahe(gray: np.ndarray, clip: float, tile: int) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(gray)


@dataclass
class Fisheye624Camera:
    """glass_config SLAM_camera fisheye624: kc = k1..k6, p1,p2, s1..s4."""

    fx: float
    fy: float
    cx: float
    cy: float
    kc: np.ndarray
    width: int
    height: int
    model: str = "fisheye624"

    @classmethod
    def from_glass_config(cls, config_path: Path, device_id: int = 1) -> "Fisheye624Camera":
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        key = f"device_{device_id}"
        slam = cfg.get("SLAM_camera", {})
        if key not in slam:
            raise KeyError(f"{config_path}: missing SLAM_camera.{key}")
        cam = slam[key]
        model = cam.get("camera_model", "")
        if model != "fisheye624":
            raise ValueError(f"unsupported camera_model={model!r}, expect fisheye624")
        kc = np.asarray(cam["kc"], dtype=np.float64).reshape(-1)
        if kc.size != 12:
            raise ValueError(f"fisheye624 kc must have 12 coeffs, got {kc.size}")
        fc = cam["fc"]
        cc = cam["cc"]
        res = cam["resolution"]
        return cls(
            fx=float(fc[0]),
            fy=float(fc[1]),
            cx=float(cc[0]),
            cy=float(cc[1]),
            kc=kc,
            width=int(res[0]),
            height=int(res[1]),
            model=model,
        )


def fisheye624_project_pixels(
    x: np.ndarray, y: np.ndarray, cam: Fisheye624Camera
) -> tuple[np.ndarray, np.ndarray]:
    k = cam.kc
    k0, k1, k2, k3, k4, k5 = k[0:6]
    p0, p1 = k[6], k[7]
    s0, s1, s2, s3 = k[8], k[9], k[10], k[11]

    r = np.sqrt(x * x + y * y)
    th = np.arctan(r)
    safe = r > 1e-12
    cos_phi = np.ones_like(r)
    sin_phi = np.zeros_like(r)
    cos_phi[safe] = x[safe] / r[safe]
    sin_phi[safe] = y[safe] / r[safe]

    th2 = th * th
    th3 = th2 * th
    th5 = th3 * th2
    th7 = th5 * th2
    th9 = th7 * th2
    th11 = th9 * th2
    th13 = th11 * th2
    th_k = th + k0 * th3 + k1 * th5 + k2 * th7 + k3 * th9 + k4 * th11 + k5 * th13
    xr = th_k * cos_phi
    yr = th_k * sin_phi

    rd2 = xr * xr + yr * yr
    rd4 = rd2 * rd2
    u = xr + (2.0 * xr * xr + rd2) * p0 + 2.0 * xr * yr * p1 + s0 * rd2 + s1 * rd4
    v = yr + (2.0 * yr * yr + rd2) * p1 + 2.0 * xr * yr * p0 + s2 * rd2 + s3 * rd4
    return cam.fx * u + cam.cx, cam.fy * v + cam.cy


def make_undistort_maps(width: int, height: int, cam: Fisheye624Camera):
    us, vs = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    x = (us - cam.cx) / cam.fx
    y = (vs - cam.cy) / cam.fy
    src_u, src_v = fisheye624_project_pixels(x, y, cam)
    return src_u.astype(np.float32), src_v.astype(np.float32)


def to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def draw_detections(bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    out = bgr.copy()
    for det in detections:
        pts = np.rint(det["corners"]).astype(np.int32)
        for k in range(4):
            cv2.line(out, tuple(pts[k]), tuple(pts[(k + 1) % 4]), (0, 255, 0), 2)
            cv2.circle(out, tuple(pts[k]), 3, (0, 0, 255), -1)
        cx, cy = det["center"]
        cv2.putText(
            out,
            str(det["id"]),
            (int(round(cx)) + 6, int(round(cy)) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def main() -> int:
    args = parse_args()

    if args.image is not None:
        images = [args.image]
    else:
        if not args.image_dir.is_dir():
            print(f"image dir not found: {args.image_dir}", file=sys.stderr)
            return 1
        images = list_images(args.image_dir)
        if not images:
            print(f"no images in {args.image_dir}", file=sys.stderr)
            return 1
        end = len(images) if args.end < 0 else min(args.end, len(images))
        images = images[args.start:end:max(1, args.stride)]

    cam = None
    if args.undistort:
        try:
            cam = Fisheye624Camera.from_glass_config(args.glass_config, args.cam_device)
        except Exception as exc:
            print(f"failed to load glass_config: {exc}", file=sys.stderr)
            return 1

    try:
        detectors = [
            EthzAprilTagDetector(
                family=args.family,
                hamming=getattr(args, "hamming", 2),
                black_border=getattr(args, "black_border", 1),
                cli_path=getattr(args, "cli", DEFAULT_CLI),
            )
        ]
        threshold_constants = [0.0]
    except Exception as exc:
        print(f"failed to initialize ethz AprilTag detector: {exc}", file=sys.stderr)
        return 1

    out_dir = args.out_dir
    csv_writer = None
    csv_file = None
    vis_dir = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        vis_dir = out_dir / "vis"
        vis_dir.mkdir(exist_ok=True)
        csv_file = open(out_dir / "detections.csv", "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "image",
                "tag_id",
                "votes",
                "passes",
                "threshold_constant",
                "perimeter_px",
                "area_px2",
                "cx",
                "cy",
                "c0_x",
                "c0_y",
                "c1_x",
                "c1_y",
                "c2_x",
                "c2_y",
                "c3_x",
                "c3_y",
            ]
        )
        csv_file.flush()

    maps = None
    n_frames = 0
    n_hit_frames = 0
    n_tags = 0
    n_rejected = 0
    paused = False
    n_total = len(images)
    progress_every = 50

    print(
        f"processing {n_total} image(s) -> {out_dir.resolve() if out_dir else '(no out)'}\n"
        f"  backend=ethz_apriltag2 family={args.family} upsample={args.upsample} "
        f"threshold={args.adaptive_thresh_min}:{args.adaptive_thresh_step}:"
        f"{args.adaptive_thresh_max} constants={threshold_constants} "
        f"clahe={args.use_clahe} "
        f"undistort={args.undistort} corner_refine={args.corner_refine}",
        flush=True,
    )

    try:
        for img_path in images:
            image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"failed to read: {img_path}", file=sys.stderr)
                continue

            gray = to_gray(image)
            if args.undistort:
                if maps is None:
                    assert cam is not None
                    h, w = gray.shape[:2]
                    print(f"  building fisheye624 undistort maps {w}x{h} ...", flush=True)
                    maps = make_undistort_maps(w, h, cam)
                    print("  undistort maps ready", flush=True)
                gray = cv2.remap(
                    gray,
                    maps[0],
                    maps[1],
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )

            gray_det = (
                apply_clahe(gray, args.clahe_clip, args.clahe_tile)
                if args.use_clahe
                else gray
            )
            all_detections: list[dict] = []
            rejected_count = 0
            for threshold_constant, detector in zip(threshold_constants, detectors):
                pass_detections, pass_rejected = detector.detect(
                    gray_det,
                    args.upsample,
                )
                for det in pass_detections:
                    det["threshold_constant"] = threshold_constant
                all_detections.extend(pass_detections)
                rejected_count += pass_rejected

            detections = merge_detections(
                all_detections,
                center_threshold=args.merge_center_threshold,
            )

            n_frames += 1
            n_rejected += rejected_count
            if detections:
                n_hit_frames += 1
                n_tags += len(detections)

            if not args.quiet:
                if detections:
                    ids = [d["id"] for d in detections]
                    print(
                        f"[{n_frames}/{n_total}] {img_path.name}: "
                        f"{len(detections)} tag(s) {ids}, rejected={rejected_count}",
                        flush=True,
                    )
                elif n_frames == 1 or n_frames % progress_every == 0:
                    print(
                        f"[{n_frames}/{n_total}] {img_path.name}: 0 tag(s) "
                        f"(hits={n_hit_frames}, rejected={rejected_count})",
                        flush=True,
                    )

            if args.only_hits and not detections:
                continue

            if csv_writer is not None:
                for det in detections:
                    c = det["corners"]
                    csv_writer.writerow(
                        [
                            img_path.name,
                            det["id"],
                            det.get("votes", 1),
                            det.get("passes", 1),
                            det.get("threshold_constant", ""),
                            f"{det['perimeter']:.3f}",
                            f"{det['area']:.3f}",
                            f"{det['center'][0]:.3f}",
                            f"{det['center'][1]:.3f}",
                            f"{c[0, 0]:.3f}",
                            f"{c[0, 1]:.3f}",
                            f"{c[1, 0]:.3f}",
                            f"{c[1, 1]:.3f}",
                            f"{c[2, 0]:.3f}",
                            f"{c[2, 1]:.3f}",
                            f"{c[3, 0]:.3f}",
                            f"{c[3, 1]:.3f}",
                        ]
                    )
                csv_file.flush()

            save_vis = vis_dir is not None and (detections or not args.only_hits)
            if save_vis or args.show:
                vis = draw_detections(to_bgr(gray), detections)
                if save_vis:
                    vis_path = vis_dir / f"{img_path.stem}.png"
                    if not cv2.imwrite(str(vis_path), vis):
                        print(f"failed to write vis: {vis_path}", file=sys.stderr)
                    elif not args.quiet:
                        print(f"  saved {vis_path}", flush=True)
                if args.show:
                    cv2.imshow("tag_detect_opencv", vis)
                    key = cv2.waitKey(0 if paused else 1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    if key == ord(" "):
                        paused = not paused
    finally:
        if csv_file is not None:
            csv_file.close()
        if args.show:
            cv2.destroyAllWindows()

    print(
        f"done: frames={n_frames}, hit_frames={n_hit_frames}, tags={n_tags}, "
        f"rejected={n_rejected}, family={args.family}, upsample={args.upsample}, "
        f"clahe={args.use_clahe}, undistort={args.undistort}"
    )
    if out_dir is not None:
        print(f"results: {out_dir.resolve()}")
        print(f"vis: {(out_dir / 'vis').resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
