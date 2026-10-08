#!/usr/bin/env python3
"""Extract Tank rosbag /imu/data into imu0.txt for stereo-inertial TagFusion.

Format (ORB-SLAM3 LoadIMU 7-column):
  # t gx gy gz ax ay az   (s, rad/s, m/s^2)

Default layout:
  <bag_dir>/../<bag_stem>/imu0.txt
  e.g. /home/geneta/下载/Tank/short_test/imu0.txt
"""

from __future__ import annotations

import argparse
import os
import sys

import rosbag

IMU_TOPIC = "/imu/data"


def stamp_str(stamp) -> str:
    return f"{stamp.to_sec():.9f}"


def extract_imu(bag_path: str, output_dir: str) -> int:
    os.makedirs(output_dir, exist_ok=True)
    imu_path = os.path.join(output_dir, "imu0.txt")
    n = 0
    print(f"[info] bag: {bag_path}")
    print(f"[info] imu: {imu_path}")

    bag = rosbag.Bag(bag_path, "r")
    with open(imu_path, "w") as f:
        f.write("# t gx gy gz ax ay az   (s, rad/s, m/s^2)\n")
        for _, msg, _t in bag.read_messages(topics=[IMU_TOPIC]):
            ts = stamp_str(msg.header.stamp)
            g = msg.angular_velocity
            a = msg.linear_acceleration
            f.write(
                f"{ts} {g.x:.12f} {g.y:.12f} {g.z:.12f} "
                f"{a.x:.12f} {a.y:.12f} {a.z:.12f}\n"
            )
            n += 1
            if n % 50000 == 0:
                print(f"[info] imu samples {n}")
    bag.close()
    print(f"[done] imu samples={n} -> {imu_path}")
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", help="path to .bag (or a directory of .bag files)")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="output directory (default: <bag_parent>/../<bag_stem>)",
    )
    args = parser.parse_args()

    bag_arg = os.path.abspath(args.bag)
    bags = []
    if os.path.isdir(bag_arg):
        bags = sorted(
            os.path.join(bag_arg, n)
            for n in os.listdir(bag_arg)
            if n.endswith(".bag")
        )
        if not bags:
            print(f"ERROR: no .bag files in {bag_arg}", file=sys.stderr)
            sys.exit(1)
    else:
        bags = [bag_arg]

    for bag_path in bags:
        stem = os.path.splitext(os.path.basename(bag_path))[0]
        if args.output is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(bag_path)), stem
            )
        elif len(bags) == 1:
            output_dir = os.path.abspath(args.output)
        else:
            output_dir = os.path.join(os.path.abspath(args.output), stem)
        extract_imu(bag_path, output_dir)


if __name__ == "__main__":
    main()
