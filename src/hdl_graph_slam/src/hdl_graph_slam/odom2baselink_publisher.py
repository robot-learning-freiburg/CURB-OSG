#!/usr/bin/python
# SPDX-License-Identifier: BSD-2-Clause
# pylint: disable=wildcard-import, attribute-defined-outside-init, unused-wildcard-import,
# pylint: disable=missing-module-docstring, missing-class-docstring, duplicate-code
import time

import rospy
import tf
from geometry_msgs.msg import *
from nav_msgs.msg import Odometry


class Odom2BaseLinkPub:
    def __init__(self):
        self.broadcaster = tf.TransformBroadcaster()
        self.agent_no = rospy.get_param("~agent_no")

        print(f"Odom2BaseLink Publisher Agent No: {self.agent_no}")
        self.subscriber = rospy.Subscriber(
            f"/robotcar_{self.agent_no}/dlio/odom_node/odom",
            Odometry,
            self.callback,
        )

        self.last_stamp = rospy.Time(0)

    def callback(self, odom_msg: Odometry):
        if self.last_stamp == odom_msg.header.stamp:
            return
        self.last_stamp = odom_msg.header.stamp
        

        transl = odom_msg.pose.pose.position
        rot = odom_msg.pose.pose.orientation
        pos = (transl.x, transl.y, transl.z)
        # dlio sends weird unnormalized quaternions sometimes
        if abs(rot.x + rot.y + rot.z) == 0.0:
            rot.w = 1.0
        quat = (rot.x, rot.y, rot.z, rot.w)

        map_frame_id = odom_msg.header.frame_id
        odom_frame_id = odom_msg.child_frame_id

        self.broadcaster.sendTransform(
            pos,
            quat,
            # self.odom_msg.header.stamp,
            odom_msg.header.stamp,
            odom_frame_id,
            map_frame_id,
        )

if __name__ == "__main__":
    rospy.init_node("odom2base_link_publisher")
    node = Odom2BaseLinkPub()
    rospy.spin()
