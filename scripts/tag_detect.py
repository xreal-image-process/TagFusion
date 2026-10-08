#!/usr/bin/env python3
"""Detect AprilTags via Thirdparty/apriltag (ethz_apriltag2).

Backend: scripts/ethz_tag_detect (C++ CLI wrapping AprilTags::TagDetector),
same library as ORB-SLAM3 ua_tag::AprilTagDetector.

Default image dir:
  /home/geneta/dataset/20260227-room-patrol-11-1/raw_data/glass/cam0/images
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
ORB_ROOT = SCRIPT_DIR.parent
DEFAULT_CLI = SCRIPT_DIR / "ethz_tag_detect"
DEFAULT_IMAGE_DIR = Path(
    "/home/geneta/dataset/20260227-room-patrol-11-1/raw_data/glass/cam0/images"
)
DEFAULT_GLASS_CONFIG = Path(
    "/home/geneta/dataset/20260227-room-patrol-11-1/raw_data/glass/glass_config.json"
)

# Families supported by ethz_apriltag2 in Thirdparty/apriltag
SUPPORTED_FAMILIES = ("tag36h11", "tag36h9", "tag25h9", "tag25h7", "tag16h5")


class AprilTagNative:
    """Python front-end for ethz_tag_detect CLI (Thirdparty/apriltag)."""

    def __init__(
        self,
        cli_path: Path = DEFAULT_CLI,
        family: str = "tag36h11",
        hamming: int = 2,
        black_border: int = 1,
        **_ignored,
    ):
        if family not in SUPPORTED_FAMILIES:
            raise ValueError(
                f"unsupported family for ethz backend: {family}; "
                f"choose from {SUPPORTED_FAMILIES}"
            )
        if not cli_path.is_file():
            raise FileNotFoundError(
                f"ethz_tag_detect CLI not found: {cli_path}\n"
                "Build it: cmake --build build --target ethz_tag_detect"
            )
        self.cli_path = cli_path
        self.family = family
        self.hamming = int(hamming)
        self.black_border = int(black_border)

    def detect(self, gray: np.ndarray) -> list[dict]:
        if gray.ndim != 2:
            raise ValueError("detect() expects grayscale uint8 image")
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cv2.imwrite(str(tmp_path), gray)
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
                err = (proc.stderr or "").strip()
                raise RuntimeError(
                    f"ethz_tag_detect failed (code={proc.returncode}): {err}"
                )
            out = []
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                corners = np.asarray(obj["corners"], dtype=np.float64)
                center = obj.get("center", corners.mean(axis=0).tolist())
                out.append(
                    {
                        "id": int(obj["id"]),
                        "hamming": int(obj["hamming"]),
                        "decision_margin": float(obj.get("perimeter", 0.0)),
                        "center": (float(center[0]), float(center[1])),
                        "corners": corners,
                    }
                )
            return out
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    def close(self) -> None:
        pass

    def __del__(self):
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect AprilTags in Glass cam0 images (libapriltag)"
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
        help="AprilTag family (ethz_apriltag2; default tag36h11)",
    )
    p.add_argument(
        "--black-border",
        type=int,
        default=1,
        help="ethz TagDetector blackBorder (configurable; standard tags: 1)",
    )
    p.add_argument(
        "--hamming",
        type=int,
        default=2,
        help="Max accepted ethz hammingDistance (default 2)",
    )
    p.add_argument(
        "--min-margin",
        type=float,
        default=0.0,
        help="Drop detections with decision_margin (perimeter) below this (default 0)",
    )
    p.add_argument(
        "--clahe",
        dest="clahe",
        action="store_true",
        default=False,
        help="Enable CLAHE (default: off for max recall)",
    )
    p.add_argument(
        "--no-clahe",
        dest="clahe",
        action="store_false",
        help="Disable CLAHE (default)",
    )
    p.add_argument("--clahe-clip", type=float, default=1.0)
    p.add_argument("--clahe-tile", type=int, default=8)
    p.add_argument(
        "--undistort",
        dest="undistort",
        action="store_true",
        default=False,
        help="Undistort with glass_config.json fisheye624 (default: off)",
    )
    p.add_argument(
        "--no-undistort",
        dest="undistort",
        action="store_false",
        help="Disable undistort (default)",
    )
    p.add_argument(
        "--glass-config",
        type=Path,
        default=DEFAULT_GLASS_CONFIG,
        help="Path to glass_config.json (fisheye624 intrinsics)",
    )
    p.add_argument(
        "--cam-device",
        type=int,
        default=1,
        help="SLAM_camera device index in glass_config (1=cam0, 2=cam1)",
    )
    p.add_argument(
        "--cli",
        type=Path,
        default=DEFAULT_CLI,
        help="Path to ethz_tag_detect CLI (Thirdparty/apriltag backend)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR / "tag_test_out",
        help="Save vis/ + detections.csv (default: scripts/tag_test_out)",
    )
    p.add_argument("--show", action="store_true", help="Window: q/ESC quit, space pause")
    p.add_argument(
        "--only-hits",
        dest="only_hits",
        action="store_true",
        default=True,
        help="Only save/show frames with detections (default)",
    )
    p.add_argument(
        "--no-only-hits",
        dest="only_hits",
        action="store_false",
        help="Save/show every processed frame",
    )
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
    kc: np.ndarray  # shape (12,)
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
        res = cam["resolution"]  # [W, H]
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
    """Project normalized pinhole rays (x=X/Z, y=Y/Z) to distorted pixels."""
    k = cam.kc
    k0, k1, k2, k3, k4, k5 = k[0:6]
    p0, p1 = k[6], k[7]
    s0, s1, s2, s3 = k[8], k[9], k[10], k[11]

    r = np.sqrt(x * x + y * y)
    th = np.arctan(r)
    # avoid /0 at optical center
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
    # tangential + thin-prism
    u = xr + (2.0 * xr * xr + rd2) * p0 + 2.0 * xr * yr * p1 + s0 * rd2 + s1 * rd4
    v = yr + (2.0 * yr * yr + rd2) * p1 + 2.0 * xr * yr * p0 + s2 * rd2 + s3 * rd4
    return cam.fx * u + cam.cx, cam.fy * v + cam.cy


def make_undistort_maps(width: int, height: int, cam: Fisheye624Camera):
    """Build remap for fisheye624 -> pinhole (same K as glass_config)."""
    # Destination (undistorted) pixel grid
    us, vs = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    x = (us - cam.cx) / cam.fx
    y = (vs - cam.cy) / cam.fy
    src_u, src_v = fisheye624_project_pixels(x, y, cam)
    map1 = src_u.astype(np.float32)
    map2 = src_v.astype(np.float32)
    return map1, map2


def to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def draw_detections(bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    out = bgr.copy()
    for det in detections:
        pts = det["corners"].astype(np.int32)
        for k in range(4):
            cv2.line(out, tuple(pts[k]), tuple(pts[(k + 1) % 4]), (0, 255, 0), 2)
            cv2.circle(out, tuple(pts[k]), 3, (0, 0, 255), -1)
        cx, cy = det["center"]
        cv2.putText(
            out,
            str(det["id"]),
            (int(cx) + 6, int(cy) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def main() -> int:
    args = parse_args()
    use_clahe = bool(args.clahe)

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
        images = images[args.start : end : max(1, args.stride)]

    cam = None
    if args.undistort:
        try:
            cam = Fisheye624Camera.from_glass_config(args.glass_config, args.cam_device)
        except Exception as e:
            print(f"failed to load glass_config: {e}", file=sys.stderr)
            return 1

    try:
        detector = AprilTagNative(
            cli_path=args.cli,
            family=args.family,
            hamming=args.hamming,
            black_border=args.black_border,
        )
    except Exception as e:
        print(f"failed to init ethz_tag_detect: {e}", file=sys.stderr)
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
                "hamming",
                "decision_margin",
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
    paused = False
    n_total = len(images)
    progress_every = 50

    print(
        f"processing {n_total} image(s) -> {out_dir.resolve() if out_dir else '(no out)'}\n"
        f"  family={args.family} hamming={args.hamming} "
        f"black_border={args.black_border} "
        f"only_hits={args.only_hits} undistort={args.undistort} "
        f"clahe={use_clahe}",
        flush=True,
    )
    if cam is not None:
        print(
            f"  glass_config={args.glass_config} device_{args.cam_device} "
            f"model={cam.model} fx={cam.fx:.3f} fy={cam.fy:.3f} "
            f"cx={cam.cx:.3f} cy={cam.cy:.3f} kc[0:4]={cam.kc[:4]}",
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
                    gray, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
                )

            gray_det = (
                apply_clahe(gray, args.clahe_clip, args.clahe_tile) if use_clahe else gray
            )
            detections = detector.detect(gray_det)
            if args.min_margin > 0:
                detections = [
                    d for d in detections if d["decision_margin"] >= args.min_margin
                ]

            n_frames += 1
            if detections:
                n_hit_frames += 1
                n_tags += len(detections)

            # Always show progress (only_hits used to skip all prints → looked hung)
            if not args.quiet:
                if detections:
                    ids = [d["id"] for d in detections]
                    print(
                        f"[{n_frames}/{n_total}] {img_path.name}: "
                        f"{len(detections)} tag(s) {ids}",
                        flush=True,
                    )
                elif n_frames == 1 or n_frames % progress_every == 0:
                    print(
                        f"[{n_frames}/{n_total}] {img_path.name}: 0 tag(s) "
                        f"(hits={n_hit_frames})",
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
                            det["hamming"],
                            f"{det['decision_margin']:.4f}",
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
                bgr = to_bgr(gray)
                vis = draw_detections(bgr, detections)
                if save_vis:
                    vis_path = vis_dir / f"{img_path.stem}.png"
                    if not cv2.imwrite(str(vis_path), vis):
                        print(f"failed to write vis: {vis_path}", file=sys.stderr)
                    elif not args.quiet:
                        print(f"  saved {vis_path}", flush=True)
                if args.show:
                    cv2.imshow("tag_detect", vis)
                    key = cv2.waitKey(0 if paused else 1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    if key == ord(" "):
                        paused = not paused
    finally:
        detector.close()
        if csv_file is not None:
            csv_file.close()
        if args.show:
            cv2.destroyAllWindows()

    print(
        f"done: frames={n_frames}, hit_frames={n_hit_frames}, tags={n_tags}, "
        f"family={args.family}, hamming={args.hamming}, "
        f"black_border={args.black_border}, clahe={use_clahe}, "
        f"undistort={args.undistort}"
    )
    if out_dir is not None:
        print(f"results: {out_dir.resolve()}")
        print(f"vis: {(out_dir / 'vis').resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
