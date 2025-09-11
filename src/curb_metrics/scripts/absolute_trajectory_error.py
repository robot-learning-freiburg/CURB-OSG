import csv
import sys
import time
from functools import partial
import os
from os import path
from token import AT
from typing import Dict, List, Tuple

import numpy as np
import ros_numpy
import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped, Point
from nav_msgs.msg import Path
from tf import Transformer, TransformListener, ExtrapolationException
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Time
from curb_metrics.msg import ATEMetric
from geometry_msgs.msg import Pose
from tf import transformations
from hdl_graph_slam.msg import Keyframe_msg, KeyframeArray_msg

# maximum duration of a run in seconds
MAX_RUN_DURATION = 3600  # 60 min

def compute_ate(keyframes: List[Keyframe_msg], transformer):
    kf: Keyframe_msg
    trans_errs = []
    tl = TransformListener()
    for kf in keyframes:
        q = kf.odom.orientation
        r = transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        t = kf.odom.position
        kf_tf = transformations.compose_matrix(translate=(t.x, t.y, t.z), angles=r)
        # kf_tf = transformations.compose_matrix(translate=(t.x, t.y, 0.0), angles=(0.0, 0.0, r[2]))
        try:
            t, r = transformer.lookupTransform(
                "world", f"robotcar_{kf.agent_no}/gt/base_link", kf.header.stamp
            )
        except ExtrapolationException as e:
            # rospy.logwarn_throttle(1.0, repr(e))
            print("oh no: ", repr(e))
            continue

        gt_tf = tl.fromTranslationRotation(translation=t, rotation=r)

        # ma.markers.append(self.get_assoc_marker(kf_tf, gt_tf))

        kf_tf: np.matrix
        diff = np.linalg.inv(gt_tf) @ kf_tf
        # store squared error
        trans_errs.append(np.linalg.norm(diff[0:3, 3]))

    if len(trans_errs) == 0:
        rospy.logwarn("Mean Absolute Trajectory Error: no samples")
        return 0.0, 0.0

    trans_errs = np.array(trans_errs)

    mate = trans_errs.mean()
    std = trans_errs.std()

    return mate, std

class TrajErrNode:
    def __init__(self):

        self.keyframes_sub = rospy.Subscriber(
            "/optimized_keyframes", KeyframeArray_msg, self.keyframes_callback
        )

        self.metric_pub = rospy.Publisher("/metric_ate", ATEMetric)

        self.tf_listener = TransformListener(
            cache_time=rospy.Duration(MAX_RUN_DURATION), interpolate=True
        )
        self.start_time = rospy.Time.now()

        # self.marker_pub = rospy.Publisher("/ate_dbg", MarkerArray)
        self.marker_id = 0

        rospy.loginfo("Absolute Trajectory Error metrics node started")


        
    def keyframes_callback(self, keyframes_array: KeyframeArray_msg) -> None:
        assert keyframes_array.keyframes is not None

        if len(keyframes_array.keyframes) == 0:
            return

        # ma = MarkerArray()
        mate, std = compute_ate(keyframes_array.keyframes, self.tf_listener)

        rospy.loginfo(f"Mean Absolute Trajectory Error: {mate:.2f} (std: {std:.3f})")

        msg = ATEMetric()
        msg.header.stamp = rospy.Time.now()
        msg.mean_ate = mate
        msg.std_ate = std
        self.metric_pub.publish(msg)

        # self.marker_pub.publish(ma)

    def get_assoc_marker(self, kf_tf, gt_tf):
        """generate arrow marker from kf to gt for validation"""
        m = Marker()

        m.header.frame_id = "world"
        m.header.stamp = rospy.Time.now()
        m.ns = "ate"
        m.id = self.marker_id
        self.marker_id += 1
        m.type = m.ARROW
        m.action = m.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 5.0
        m.scale.y = 7.0
        m.color.a = 1.0
        m.color.b = 0.8
        m.color.g = 0.2
        p_a = Point()
        p_a.x = kf_tf[0, 3]
        p_a.y = kf_tf[1, 3]
        p_a.z = kf_tf[2, 3]

        p_b = Point()
        p_b.x = gt_tf[0, 3]
        p_b.y = gt_tf[1, 3]
        p_b.z = gt_tf[2, 3]

        m.points = [p_a, p_b]

        return m


if __name__ == "__main__":
    rospy.init_node("trajectory_error_node")
    rmse_node = TrajErrNode()
    rospy.spin()
