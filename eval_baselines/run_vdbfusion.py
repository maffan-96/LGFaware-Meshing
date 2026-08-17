#!/usr/bin/env python3
"""VDBFusion TSDF baseline using the SAME dataset conventions as qem_voxel_map.

Data layout (mirrors qem_voxel_map.cpp / qem_mesh.cpp exactly):
  - PCD point clouds (binary or ASCII), one file per scan, points in the SENSOR frame
  - Pose file in TUM, G2O/SLAM, or CSV format, inferred from the extension:
        .tum          -> TUM   (also the default for unknown extensions)
        .g2o / .slam  -> G2O   (VERTEX_SE3:QUAT_TIME id x y z qx qy qz qw sec nsec)
        .csv          -> CSV   (counter, sec, nsec, x, y, z, qx, qy, qz, qw)
  - PCD filenames matched to the pose key '%010d_%09d' by trying every adjacent
    (sec, nsec) digit-token pair of the stem from the right; first match wins

Per-scan pipeline (mirrors process_scan):
  1. Read PCD -> finite sensor-frame points
  2. Subsample with --process_every_n (points[::n])
  3. Range window: drop points with |p_sensor| outside [min_point_range, max_point_range]
  4. Transform to the global frame: p_g = R @ p_s + t; sensor origin = t
  5. Optional axis-aligned ROI in the reconstruction frame (drop before ingest)
  6. TSDF integrate(points_global, origin) -- VDBFusion's 4x4 overload does NOT
     transform points (it only reads the translation), so the transform is done
     here, exactly like the reference.

Scan windowing: --start_scan N skips the first N matched scans, --num_scans M
then limits to M, so the two compose into the same window as the reference.

Setup (Ubuntu 20.04, python3.8):   pip3 install vdbfusion numpy

Comparable run against the qem_voxel_map example command:
  python3 run_vdbfusion.py \
      --pcd_folder .../undist-clouds/ --pose_file .../slam-poses.csv \
      --mesh_output mesh_vdbfusion.ply \
      --voxel_size 0.1 --sdf_trunc 0.3 --process_every_n 1 --min_weight 1.0
(voxel_size matches their voxel_size, space carving is on by default like their
ray carving, min_weight 1.0 corresponds to --min_hit_count_vertex 1.)
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ poses ----

def quat_to_T(x, y, z, qx, qy, qz, qw):
    T = np.eye(4)
    T[0, 3], T[1, 3], T[2, 3] = x, y, z
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return T                       # identity rotation, like the reference
    s = 1.0 / np.sqrt(n)
    qx, qy, qz, qw = qx * s, qy * s, qz * s, qw * s
    T[0, 0] = 1 - 2 * (qy * qy + qz * qz)
    T[0, 1] = 2 * (qx * qy - qz * qw)
    T[0, 2] = 2 * (qx * qz + qy * qw)
    T[1, 0] = 2 * (qx * qy + qz * qw)
    T[1, 1] = 1 - 2 * (qx * qx + qz * qz)
    T[1, 2] = 2 * (qy * qz - qx * qw)
    T[2, 0] = 2 * (qx * qz - qy * qw)
    T[2, 1] = 2 * (qy * qz + qx * qw)
    T[2, 2] = 1 - 2 * (qx * qx + qy * qy)
    return T


def pose_key(sec, nsec):
    return "%010d_%09d" % (sec, nsec)


def parse_poses_tum(path):
    poses = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 8:
            continue
        ts = p[0]
        if "." not in ts:
            continue
        sec_s, nsec_s = ts.split(".", 1)
        # Pad/truncate nsec to 9 digits (TUM is loose about this), like the reference.
        nsec_s = (nsec_s + "0" * 9)[:9]
        try:
            sec, nsec = int(sec_s), int(nsec_s)
            vals = [float(v) for v in p[1:8]]
        except ValueError:
            continue
        poses.append((pose_key(sec, nsec), quat_to_T(*vals)))
    return poses


def parse_poses_g2o(path):
    poses = []
    for line in open(path):
        p = line.split()
        if len(p) < 11 or p[0] != "VERTEX_SE3:QUAT_TIME":
            continue
        try:
            vals = [float(v) for v in p[2:9]]
            sec, nsec = int(p[9]), int(p[10])
        except ValueError:
            continue
        poses.append((pose_key(sec, nsec), quat_to_T(*vals)))
    return poses


def parse_poses_csv(path):
    poses = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.replace(",", " ").split()
        if len(p) < 10:
            continue
        try:
            sec, nsec = int(p[1]), int(p[2])
            vals = [float(v) for v in p[3:10]]
        except ValueError:
            continue
        poses.append((pose_key(sec, nsec), quat_to_T(*vals)))
    return poses


def load_dataset(pcd_folder, pose_file):
    """Pair PCDs to poses by filename timestamp, like the reference load_dataset."""
    pcds = sorted(str(p) for p in Path(pcd_folder).iterdir() if p.suffix == ".pcd")

    ext = Path(pose_file).suffix.lower()
    if ext in (".g2o", ".slam"):
        raw, fmt = parse_poses_g2o(pose_file), "G2O"
    elif ext == ".csv":
        raw, fmt = parse_poses_csv(pose_file), "CSV"
    else:
        raw, fmt = parse_poses_tum(pose_file), "TUM"
    print("  Parsed %d poses (%s)" % (len(raw), fmt))
    pose_map = dict(raw)

    dataset = []
    digits = re.compile(r"^\d+$")
    for pcd in pcds:
        tokens = [t for t in Path(pcd).stem.split("_") if t]
        # Try every adjacent (sec, nsec) pair from the right; first match wins.
        for i in range(len(tokens) - 1, 0, -1):
            if not (digits.match(tokens[i - 1]) and digits.match(tokens[i])):
                continue
            try:
                key = pose_key(int(tokens[i - 1]), int(tokens[i]))
            except ValueError:
                continue
            T = pose_map.get(key)
            if T is not None:
                dataset.append((pcd, T))
                break
    print("  Matched %d / %d PCDs to poses" % (len(dataset), len(pcds)))
    return dataset


# ------------------------------------------------------------------- PCD -----

def read_pcd_xyz(path):
    """PCD reader (binary + ASCII, COUNT-aware offsets) -> finite Nx3 float64."""
    with open(path, "rb") as f:
        data = f.read()

    lines, off = [], 0
    while True:
        nl = data.index(b"\n", off)
        line = data[off:nl].decode("ascii", "replace").strip()
        off = nl + 1
        if line and not line.startswith("#"):
            lines.append(line)
        if line.startswith("DATA"):
            break
    hdr = {}
    for l in lines:
        p = l.split()
        hdr[p[0]] = p[1:]
    fields = hdr.get("FIELDS", [])
    sizes = [int(x) for x in hdr.get("SIZE", ["4"] * len(fields))]
    counts = [int(x) for x in hdr.get("COUNT", ["1"] * len(fields))]
    npts = int(hdr.get("POINTS", hdr.get("WIDTH", ["0"]))[0])
    mode = hdr["DATA"][0]
    if mode == "binary_compressed":
        raise ValueError(f"{path}: binary_compressed not supported")

    offsets, cur = {}, 0
    for f_, s, c in zip(fields, sizes, counts):
        offsets[f_.lower()] = cur
        cur += s * c
    stride = cur
    for ax in ("x", "y", "z"):
        if ax not in offsets:
            raise ValueError(f"{path}: field '{ax}' missing")

    if mode == "binary":
        raw = np.frombuffer(data, dtype=np.uint8, count=npts * stride,
                            offset=off).reshape(npts, stride)
        pts = np.empty((npts, 3))
        for i, ax in enumerate(("x", "y", "z")):
            o = offsets[ax]
            pts[:, i] = raw[:, o:o + 4].copy().view("<f4").ravel()
    else:  # ascii: column index by field position, like the reference
        col = {f_.lower(): i for i, f_ in enumerate(fields)}
        arr = np.loadtxt(path, skiprows=len(lines), max_rows=npts, ndmin=2)
        pts = arr[:, [col["x"], col["y"], col["z"]]].astype(np.float64)

    return pts[np.isfinite(pts).all(axis=1)]


# ------------------------------------------------------------------- PLY -----

def write_ply(path, verts, tris, comments):
    verts = np.asarray(verts, dtype="<f4")
    tris = np.asarray(tris, dtype="<i4").reshape(-1, 3)
    with open(path, "wb") as f:
        hdr = ["ply", "format binary_little_endian 1.0"]
        hdr += ["comment " + c for c in comments]
        hdr += [f"element vertex {len(verts)}",
                "property float x", "property float y", "property float z",
                f"element face {len(tris)}",
                "property list uchar int vertex_indices", "end_header"]
        f.write(("\n".join(hdr) + "\n").encode())
        f.write(verts.tobytes())
        face = np.empty(len(tris), dtype=[("n", "u1"), ("v", "<i4", (3,))])
        face["n"] = 3
        face["v"] = tris
        f.write(face.tobytes())


# ------------------------------------------------------------------- main ----

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcd_folder", required=True)
    ap.add_argument("--pose_file", required=True)
    ap.add_argument("--mesh_output", required=True)
    # Discretisation, comparable to the reference defaults.
    ap.add_argument("--voxel_size", type=float, default=0.1)
    ap.add_argument("--sdf_trunc", type=float, default=0.3,
                    help="TSDF truncation, metres (default 3*voxel_size)")
    ap.add_argument("--no_space_carving", action="store_true",
                    help="disable free-space carving (reference: --no_carve)")
    # Ingest window, same semantics as the reference.
    ap.add_argument("--process_every_n", type=int, default=1)
    ap.add_argument("--min_point_range", type=float, default=0.0)
    ap.add_argument("--max_point_range", type=float, default=0.0)
    ap.add_argument("--roi_min", type=float, nargs=3, default=None)
    ap.add_argument("--roi_max", type=float, nargs=3, default=None)
    ap.add_argument("--start_scan", type=int, default=0)
    ap.add_argument("--num_scans", type=int, default=-1)
    # Mesh extraction.
    ap.add_argument("--min_weight", type=float, default=1.0,
                    help="min TSDF weight per vertex (~ --min_hit_count_vertex)")
    ap.add_argument("--fill_holes", action="store_true")
    args = ap.parse_args(argv)

    dataset = load_dataset(args.pcd_folder, args.pose_file)
    if not dataset:
        sys.exit("No PCDs matched to poses; nothing to do.")
    if args.start_scan > 0:
        if args.start_scan >= len(dataset):
            sys.exit(f"--start_scan {args.start_scan} skips all {len(dataset)} matched scans")
        dataset = dataset[args.start_scan:]
        print(f"Skipped the first {args.start_scan} scans (--start_scan)")
    if 0 < args.num_scans < len(dataset):
        dataset = dataset[:args.num_scans]
    print(f"Processing {len(dataset)} scans")

    roi = None
    if args.roi_min is not None and args.roi_max is not None:
        roi = (np.array(args.roi_min), np.array(args.roi_max))

    import vdbfusion  # deferred so the IO above is testable without it
    vol = vdbfusion.VDBVolume(voxel_size=args.voxel_size, sdf_trunc=args.sdf_trunc,
                              space_carving=not args.no_space_carving)
    print(f"[vdbfusion] voxel_size={args.voxel_size} sdf_trunc={args.sdf_trunc} "
          f"space_carving={not args.no_space_carving}")

    t_all = time.time()
    step = max(1, args.process_every_n)
    for i, (pcd_path, T) in enumerate(dataset):
        t0 = time.time()
        pts_s = read_pcd_xyz(pcd_path)          # finite, sensor frame
        n_in = len(pts_s)
        pts_s = pts_s[::step]
        n_used = len(pts_s)

        rng = np.linalg.norm(pts_s, axis=1)     # |p_s| == |p_g - t|
        keep = rng > 1e-9
        if args.max_point_range > 0.0:
            keep &= rng <= args.max_point_range
        if args.min_point_range > 0.0:
            keep &= rng >= args.min_point_range
        n_rangerej = int(n_used - keep.sum())
        pts_s = pts_s[keep]

        R, t = T[:3, :3], T[:3, 3]
        pts_g = pts_s @ R.T + t                 # global frame, like the reference
        if roi is not None:
            inroi = ((pts_g >= roi[0]) & (pts_g <= roi[1])).all(axis=1)
            n_rangerej += int(len(pts_g) - inroi.sum())
            pts_g = pts_g[inroi]

        if len(pts_g):
            vol.integrate(np.ascontiguousarray(pts_g), np.ascontiguousarray(t))
        print("  Scan %4d: in=%6d used=%6d kept=%6d rangerej=%5d  %.2fs  %s"
              % (i, n_in, n_used, len(pts_g), n_rangerej, time.time() - t0,
                 Path(pcd_path).name))

    print(f"Integration: {time.time() - t_all:.1f}s")
    verts, tris = vol.extract_triangle_mesh(fill_holes=args.fill_holes,
                                            min_weight=args.min_weight)
    print(f"mesh: {len(verts)} vertices, {len(tris)} triangles")
    write_ply(args.mesh_output, verts, tris, [
        "vdbfusion TSDF baseline (qem_voxel_map-compatible dataset conventions)",
        f"voxel_size {args.voxel_size}", f"sdf_trunc {args.sdf_trunc}",
        f"space_carving {not args.no_space_carving}",
        f"scans {len(dataset)}", f"min_weight {args.min_weight}",
    ])
    print(f"wrote {args.mesh_output}")


if __name__ == "__main__":
    main()
