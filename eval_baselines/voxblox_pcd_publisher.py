#!/usr/bin/env python3
"""Feed a Super-LIO / qem_voxel_map-style PCD+pose dataset into voxblox offline.

Loads the dataset exactly like run_vdbfusion.py (same pose formats, same
filename-to-pose matching, same windowing), then publishes per scan:

    /voxblox_input/pointcloud   sensor_msgs/PointCloud2   (sensor frame, as stored)
    /voxblox_input/transform    geometry_msgs/TransformStamped  (T_world_sensor)

both with the same stamp, which is voxblox's input contract with
use_tf_transforms: false. When all scans are published it calls the
generate_mesh service so tsdf_server writes its mesh_filename, then exits.

Run with the SYSTEM python3 (ROS Noetic), NOT inside a conda env:

    roslaunch online_mesh voxblox_superlio.launch \
        voxel_size:=0.1 mesh_filename:=$HOME/mesh_voxblox_itc_stairs.ply &
    python3 eval_baselines/voxblox_pcd_publisher.py \
        --pcd_folder .../super-lio/undist-cloud-normals/ \
        --pose_file  .../super-lio/slam_poses.csv \
        --rate 5.0
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_vdbfusion import load_dataset, read_pcd_xyz  # noqa: E402  (same conventions)

import rospy                                            # noqa: E402
from geometry_msgs.msg import TransformStamped          # noqa: E402
from sensor_msgs.msg import PointCloud2                 # noqa: E402
from sensor_msgs import point_cloud2 as pc2             # noqa: E402
from std_msgs.msg import Header                         # noqa: E402
from std_srvs.srv import Empty                          # noqa: E402


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
    ap.add_argument("--rate", type=float, default=5.0,
                    help="scans per second published (voxblox must keep up)")
    ap.add_argument("--world_frame", default="world")
    ap.add_argument("--sensor_frame", default="sensor")
    ap.add_argument("--mesh_service", default="/voxblox_node/generate_mesh")
    ap.add_argument("--no_mesh_service", action="store_true",
                    help="do not call generate_mesh when finished")
    args = ap.parse_args(rospy.myargv()[1:])

    dataset = load_dataset(args.pcd_folder, args.pose_file)
    if not dataset:
        sys.exit("No PCDs matched to poses; nothing to do.")
    if args.start_scan > 0:
        dataset = dataset[args.start_scan:]
        print(f"Skipped the first {args.start_scan} scans (--start_scan)")
    if 0 < args.num_scans < len(dataset):
        dataset = dataset[:args.num_scans]
    print(f"Publishing {len(dataset)} scans at {args.rate} Hz")

    rospy.init_node("voxblox_pcd_publisher")
    pub_cloud = rospy.Publisher("/voxblox_input/pointcloud", PointCloud2, queue_size=20)
    pub_tf = rospy.Publisher("/voxblox_input/transform", TransformStamped, queue_size=20)

    # Wait until voxblox has actually subscribed, or the first scans vanish.
    t0 = rospy.Time.now()
    while (pub_cloud.get_num_connections() == 0 or pub_tf.get_num_connections() == 0):
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
        pts = read_pcd_xyz(pcd_path)[::step].astype(np.float32)   # sensor frame

        stamp = rospy.Time.now()
        R, t = T[:3, :3], T[:3, 3]
        qx, qy, qz, qw = rot_to_quat(R)
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = args.world_frame
        tf.child_frame_id = args.sensor_frame
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = t
        tf.transform.rotation.x, tf.transform.rotation.y = qx, qy
        tf.transform.rotation.z, tf.transform.rotation.w = qz, qw

        header = Header(stamp=stamp, frame_id=args.sensor_frame)
        cloud = pc2.create_cloud_xyz32(header, pts)

        pub_tf.publish(tf)      # transform first so voxblox can match the cloud stamp
        pub_cloud.publish(cloud)
        if i % 25 == 0 or i == len(dataset) - 1:
            print(f"  [{i + 1}/{len(dataset)}] {os.path.basename(pcd_path)}  pts={len(pts)}")
        rate.sleep()

    # Let voxblox drain its queue before asking for the mesh.
    print("All scans published; waiting 5 s for voxblox to finish integrating ...")
    rospy.sleep(5.0)
    if not args.no_mesh_service:
        print(f"Calling {args.mesh_service} ...")
        rospy.wait_for_service(args.mesh_service, timeout=30.0)
        rospy.ServiceProxy(args.mesh_service, Empty)()
        print("generate_mesh done -- voxblox wrote its mesh_filename.")


if __name__ == "__main__":
    main()
