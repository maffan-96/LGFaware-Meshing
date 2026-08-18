#!/usr/bin/env python3
"""Feed a PCD+pose dataset (Super-LIO / VILENS export) into the LGFaware
meshing node offline.

Loads the dataset exactly like run_vdbfusion.py (same pose formats, same
filename-to-pose matching, same windowing), then publishes per scan what
the meshing node configured with mid360_superlio.yaml expects:

    /lio/cloud_world   sensor_msgs/PointCloud2 (XYZI, WORLD frame)
    /lio/odom          nav_msgs/Odometry       (T_world_sensor)

both with the same stamp (the node pairs them with sync_eps = 0.01 s).
The stored PCDs are in the sensor frame, so each scan is transformed by
its pose before publishing.

When all scans are published this script exits; the meshing node keeps
running until YOU Ctrl-C it, which is what triggers its mesh saving
(p.ply / np.ply / all.ply under ptcl_save_path/<dataset_name>_<time>/).

Run (each terminal: conda activate ros_noetic + source the LGFaware ws):

    roslaunch online_mesh meshing_only_superlio.launch     # terminal 1
    python3 lgfaware_pcd_publisher.py \
        --pcd_folder .../undist-clouds/ --pose_file .../slam-poses.csv \
        --rate 2.0                                          # terminal 2
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_vdbfusion import load_dataset, read_pcd_xyz  # noqa: E402  (same conventions)

import rospy                                            # noqa: E402
from nav_msgs.msg import Odometry                       # noqa: E402
from sensor_msgs.msg import PointCloud2, PointField     # noqa: E402
from sensor_msgs import point_cloud2 as pc2             # noqa: E402
from std_msgs.msg import Header                         # noqa: E402

XYZI_FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
]


def rot_to_quat(R):
    """Rotation matrix -> (x, y, z, w), Shepperd's method."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcd_folder", required=True)
    ap.add_argument("--pose_file", required=True)
    ap.add_argument("--process_every_n", type=int, default=1)
    ap.add_argument("--start_scan", type=int, default=0)
    ap.add_argument("--num_scans", type=int, default=-1)
    ap.add_argument("--rate", type=float, default=2.0,
                    help="scans per second published (meshing must keep up)")
    ap.add_argument("--cloud_topic", default="/lio/cloud_world")
    ap.add_argument("--odom_topic", default="/lio/odom")
    ap.add_argument("--world_frame", default="camera_init")
    args = ap.parse_args(rospy.myargv()[1:])

    dataset = load_dataset(args.pcd_folder, args.pose_file)
    if not dataset:
        sys.exit("No PCDs matched to poses; nothing to do.")
    if args.start_scan > 0:
        dataset = dataset[args.start_scan:]
        print(f"Skipped the first {args.start_scan} scans (--start_scan)")
    if 0 < args.num_scans < len(dataset):
        dataset = dataset[:args.num_scans]
    print(f"Publishing {len(dataset)} scans at {args.rate} Hz "
          f"on {args.cloud_topic} (world frame) + {args.odom_topic}")

    rospy.init_node("lgfaware_pcd_publisher")
    pub_cloud = rospy.Publisher(args.cloud_topic, PointCloud2, queue_size=20)
    pub_odom = rospy.Publisher(args.odom_topic, Odometry, queue_size=20)

    # Wait until the meshing node has subscribed, or the first scans vanish.
    t0 = rospy.Time.now()
    while (pub_cloud.get_num_connections() == 0 or pub_odom.get_num_connections() == 0):
        if rospy.is_shutdown():
            return
        if (rospy.Time.now() - t0).to_sec() > 30.0:
            print("WARNING: no subscriber after 30 s; publishing anyway")
            break
        rospy.sleep(0.1)

    rate = rospy.Rate(max(0.1, args.rate))
    step = max(1, args.process_every_n)
    for i, (pcd_path, T) in enumerate(dataset):
        if rospy.is_shutdown():
            return
        pts_s = read_pcd_xyz(pcd_path)[::step]                 # sensor frame
        R, t = T[:3, :3], T[:3, 3]
        pts_w = (pts_s @ R.T + t).astype(np.float32)           # world frame
        xyzi = np.column_stack([pts_w, np.zeros(len(pts_w), np.float32)])

        stamp = rospy.Time.now()
        qx, qy, qz, qw = rot_to_quat(R)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = args.world_frame
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = t
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        header = Header(stamp=stamp, frame_id=args.world_frame)
        cloud = pc2.create_cloud(header, XYZI_FIELDS, xyzi)

        pub_odom.publish(odom)   # odometry first; the node waits for the pose
        rospy.sleep(0.02)
        pub_cloud.publish(cloud)
        if i % 25 == 0 or i == len(dataset) - 1:
            print(f"  [{i + 1}/{len(dataset)}] {os.path.basename(pcd_path)}  pts={len(pts_w)}")
        rate.sleep()

    print("All scans published. Give the meshing node time to drain its "
          "buffers (watch its frame counter), then Ctrl-C the MESHING node "
          "-- that triggers its mesh saving (p.ply / np.ply / all.ply).")


if __name__ == "__main__":
    main()
