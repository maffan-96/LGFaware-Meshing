#!/usr/bin/env python3
"""Feed voxblox / vdbfusion_ros from a LIO front-end.

Both consume a sensor-frame PointCloud2 plus a geometry_msgs/TransformStamped
(T_world_sensor) matched by timestamp. LIO front-ends publish either a
body-frame cloud (FAST-LIO /cloud_registered_body) or a world-frame cloud
(Super-LIO /lio/cloud_world) alongside nav_msgs/Odometry. This node pairs the
two by stamp and republishes:

    ~cloud_out      sensor_msgs/PointCloud2   (sensor frame)
    ~transform_out  geometry_msgs/TransformStamped

FAST-LIO:
    rosrun online_mesh odom_cloud_relay.py _cloud_topic:=/cloud_registered_body \
        _odom_topic:=/Odometry _cloud_in_world_frame:=false
Super-LIO:
    rosrun online_mesh odom_cloud_relay.py _cloud_topic:=/lio/cloud_world \
        _odom_topic:=/lio/odom _cloud_in_world_frame:=true
"""
from collections import deque

import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2


def quat_to_R(qx, qy, qz, qw):
    n = np.linalg.norm([qx, qy, qz, qw])
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)]])


class Relay(object):
    def __init__(self):
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.sensor_frame = rospy.get_param("~sensor_frame", "sensor")
        self.cloud_in_world = rospy.get_param("~cloud_in_world_frame", False)
        self.eps = rospy.get_param("~sync_eps", 0.01)

        self.clouds = deque()
        self.odoms = deque()

        self.pub_cloud = rospy.Publisher("~cloud_out", PointCloud2, queue_size=10)
        self.pub_tf = rospy.Publisher("~transform_out", TransformStamped, queue_size=10)
        rospy.Subscriber(rospy.get_param("~cloud_topic"), PointCloud2, self.cb_cloud, queue_size=50)
        rospy.Subscriber(rospy.get_param("~odom_topic"), Odometry, self.cb_odom, queue_size=200)

    def cb_cloud(self, msg):
        self.clouds.append(msg)
        self.try_match()

    def cb_odom(self, msg):
        self.odoms.append(msg)
        self.try_match()

    def try_match(self):
        while self.clouds and self.odoms:
            tc = self.clouds[0].header.stamp.to_sec()
            to = self.odoms[0].header.stamp.to_sec()
            if to < tc - self.eps:
                self.odoms.popleft()
            elif tc < to - self.eps:
                self.clouds.popleft()
                rospy.logwarn_throttle(5.0, "relay: dropped a cloud without matching odometry")
            else:
                self.emit(self.clouds.popleft(), self.odoms.popleft())

    def emit(self, cloud, odom):
        p, q = odom.pose.pose.position, odom.pose.pose.orientation

        if self.cloud_in_world:
            cloud = self.world_to_sensor(cloud, p, q)

        cloud.header.frame_id = self.sensor_frame
        tf = TransformStamped()
        tf.header.stamp = cloud.header.stamp
        tf.header.frame_id = self.world_frame
        tf.child_frame_id = self.sensor_frame
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = p.x, p.y, p.z
        tf.transform.rotation = q

        self.pub_tf.publish(tf)
        self.pub_cloud.publish(cloud)

    def world_to_sensor(self, cloud, p, q):
        """p_sensor = R^T (p_world - t), rewriting x/y/z in place, other fields kept."""
        R = quat_to_R(q.x, q.y, q.z, q.w)
        t = np.array([p.x, p.y, p.z])

        buf = bytearray(cloud.data)
        raw = np.frombuffer(buf, dtype=np.uint8).reshape(-1, cloud.point_step)
        offs = {f.name: f.offset for f in cloud.fields}
        xyz = np.empty((len(raw), 3))
        for i, ax in enumerate(("x", "y", "z")):
            xyz[:, i] = raw[:, offs[ax]:offs[ax] + 4].copy().view("<f4").ravel()
        xyz = (xyz - t) @ R                     # (R^T (p - t))^T for row vectors
        for i, ax in enumerate(("x", "y", "z")):
            raw[:, offs[ax]:offs[ax] + 4] = xyz[:, i].astype("<f4").view(np.uint8).reshape(-1, 4)
        cloud.data = bytes(buf)
        return cloud


if __name__ == "__main__":
    rospy.init_node("odom_cloud_relay")
    Relay()
    rospy.spin()
