#!/usr/bin/env python3
"""Offline VDBFusion baseline from Super-LIO's LiDAR-frame export.

Input is what Super-LIO writes with export_lidar_frame: true —
  <map_dir>/PCD/scans_<sec>_<nsec>.pcd   one PCD per scan, points in the LiDAR frame
  <map_dir>/slam_poses.csv               "# counter, sec, nsec, x, y, z, qx, qy, qz, qw"
                                         with pose = T_world_lidar, keyed on (sec, nsec)

This matches vdbfusion's integrate(points_in_sensor_frame, T_world_sensor) exactly.

Setup (Ubuntu 20.04, python3.8):
    pip3 install vdbfusion numpy

Run:
    python3 run_vdbfusion_superlio.py <map_dir> -o mesh_vdbfusion.ply \
        --voxel-size 0.1 --sdf-trunc 0.3 --space-carving
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def read_pcd_xyz(path):
    """Minimal PCD reader (ascii / binary) returning Nx3 float64 xyz."""
    with open(path, "rb") as f:
        data = f.read()
    lines = []
    off = 0
    while True:
        nl = data.index(b"\n", off)
        line = data[off:nl].decode("ascii", "replace")
        off = nl + 1
        lines.append(line)
        if line.startswith("DATA"):
            break
    hdr = {l.split()[0]: l.split()[1:] for l in lines if l and not l.startswith("#")}
    fields, sizes, types = hdr["FIELDS"], [int(s) for s in hdr["SIZE"]], hdr["TYPE"]
    counts = [int(c) for c in hdr.get("COUNT", ["1"] * len(fields))]
    npts = int(hdr["POINTS"][0])
    mode = hdr["DATA"][0]

    offsets, cur = {}, 0
    for f_, s, c in zip(fields, sizes, counts):
        offsets[f_] = cur
        cur += s * c
    stride = cur

    if mode == "ascii":
        cols = {f_: i for i, f_ in enumerate(fields)}
        arr = np.loadtxt(path, skiprows=len(lines), max_rows=npts, dtype=np.float64)
        return arr[:, [cols["x"], cols["y"], cols["z"]]]
    if mode == "binary":
        raw = np.frombuffer(data, dtype=np.uint8, count=npts * stride, offset=off).reshape(npts, stride)
        out = np.empty((npts, 3))
        for i, ax in enumerate(("x", "y", "z")):
            o = offsets[ax]
            out[:, i] = raw[:, o:o + 4].copy().view("<f4").ravel()
        return out
    raise ValueError(f"{path}: DATA {mode} not supported (binary_compressed: resave or "
                     f"set a plain-binary writer)")


def quat_to_R(qx, qy, qz, qw):
    n = np.linalg.norm([qx, qy, qz, qw])
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)]])


def load_poses(csv_path):
    """(sec, nsec) -> 4x4 T_world_lidar."""
    poses = {}
    for line in open(csv_path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        v = [p.strip() for p in line.split(",")]
        sec, nsec = int(v[1]), int(v[2])
        x, y, z, qx, qy, qz, qw = map(float, v[3:10])
        T = np.eye(4)
        T[:3, :3] = quat_to_R(qx, qy, qz, qw)
        T[:3, 3] = [x, y, z]
        poses[(sec, nsec)] = T
    return poses


def write_ply(path, verts, tris):
    verts = np.asarray(verts, dtype="<f4")
    tris = np.asarray(tris, dtype="<i4")
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {len(verts)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 f"element face {len(tris)}\n"
                 "property list uchar int vertex_indices\nend_header\n").encode())
        f.write(verts.tobytes())
        face = np.empty(len(tris), dtype=[("n", "u1"), ("v", "<i4", (3,))])
        face["n"] = 3
        face["v"] = tris
        f.write(face.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_dir", type=Path, help="Super-LIO save_map_dir (contains PCD/ and slam_poses.csv)")
    ap.add_argument("-o", "--output", type=Path, default=Path("mesh_vdbfusion.ply"))
    ap.add_argument("--voxel-size", type=float, default=0.1)
    ap.add_argument("--sdf-trunc", type=float, default=0.3)
    ap.add_argument("--space-carving", action="store_true")
    ap.add_argument("--min-weight", type=float, default=5.0)
    ap.add_argument("--fill-holes", action="store_true")
    ap.add_argument("--min-range", type=float, default=0.5, help="drop points closer than this (m)")
    ap.add_argument("--max-range", type=float, default=60.0, help="drop points farther than this (m)")
    args = ap.parse_args()

    poses = load_poses(args.map_dir / "slam_poses.csv")
    pcds = sorted((args.map_dir / "PCD").glob("scans_*.pcd"))
    print(f"{len(pcds)} scans, {len(poses)} poses in {args.map_dir}")
    if not pcds:
        sys.exit("no PCD files found — was export_lidar_frame: true and save_interval > 0 ?")

    import vdbfusion  # deferred so the IO above is testable without it
    vol = vdbfusion.VDBVolume(voxel_size=args.voxel_size, sdf_trunc=args.sdf_trunc,
                              space_carving=args.space_carving)

    used = skipped = 0
    for p in pcds:
        parts = p.stem.split("_")            # scans_<sec>_<nsec>
        key = (int(parts[1]), int(parts[2]))
        T = poses.get(key)
        if T is None:
            skipped += 1
            continue
        pts = read_pcd_xyz(p)
        pts = pts[np.isfinite(pts).all(1)]
        r = np.linalg.norm(pts, axis=1)      # sensor frame: range is just the norm
        pts = pts[(r > args.min_range) & (r < args.max_range)]
        if len(pts):
            vol.integrate(pts, T)
        used += 1
        if used % 50 == 0:
            print(f"  integrated {used}/{len(pcds)}")
    print(f"integrated {used} scans ({skipped} without pose)")

    verts, tris = vol.extract_triangle_mesh(fill_holes=args.fill_holes, min_weight=args.min_weight)
    print(f"mesh: {len(verts)} vertices, {len(tris)} triangles")
    write_ply(args.output, verts, tris)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
