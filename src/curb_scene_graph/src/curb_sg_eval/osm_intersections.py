#!/usr/bin/env python

import os
from typing import Union
import pickle
import ros_numpy
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree
from curb_scene_graph import RoadGraph, Intersection # type: ignore
from curb_metrics.msg import IntersectionMetric
from .utils import plot_intersections, plot_ways


class IntersectionsEval:
    def __init__(self, road_graph: RoadGraph):
        dataset_dir: str = rospy.get_param("~dataset_dir")  # type: ignore
        self.import_data(dataset_dir)

        self.marker_pub = rospy.Publisher(
            "osm_intersections", MarkerArray, latch=True, queue_size=10
        )

        self.metric_pub = rospy.Publisher(
            "/metric_intersections", IntersectionMetric, latch=True, queue_size=10
        )

        all_isect_color: List = rospy.get_param("~curb/osm_gt/all_intersection_color")  # type: ignore
        selected_isect_color: List = rospy.get_param("~curb/osm_gt/selected_intersection_color")  # type: ignore
        isect_scale: float = rospy.get_param("~curb/osm_gt/intersection_scale")  # type: ignore
        ways_color: float = rospy.get_param("~curb/osm_gt/ways_color")  # type: ignore
        ways_scale: float = rospy.get_param("~curb/osm_gt/ways_scale")  # type: ignore
        all_isect_z: float = rospy.get_param("~curb/osm_gt/all_isect_z")  # type: ignore
        selected_isect_z: float = rospy.get_param("~curb/osm_gt/selected_isect_z")  # type: ignore
        ways_z: float = rospy.get_param("~curb/osm_gt/ways_z")  # type: ignore

        ma = plot_intersections(self.isect_gt, self.marker_pub, "all_isects", color=all_isect_color, size=isect_scale, z_offset=all_isect_z)
        ma.markers.extend(plot_intersections(self.selected_isect_gt, self.marker_pub, "selected_isects", color=selected_isect_color, size=isect_scale, z_offset=selected_isect_z).markers)  # type: ignore
        ma.markers.extend(plot_ways(self.ways, self.marker_pub, z_offset=ways_z, color=ways_color, size=ways_scale).markers)  # type: ignore
        self.marker_pub.publish(ma)

        self.road_graph = road_graph
        self.t = rospy.Timer(rospy.Duration(4), lambda _: self.run_eval(road_graph))

        rospy.loginfo("Intersections eval ready, will run automatically")

    def import_data(self, dataset_dir):
        assert os.path.exists(dataset_dir), "Dataset directory does not exist"
        rospy.loginfo(f"Importing from {dataset_dir}")

        ways_pickle_path = os.path.join(dataset_dir, "ways.pkl")
        with open(ways_pickle_path, "rb") as f:
            self.ways = pickle.load(f)
        rospy.loginfo(f"Imported {len(self.ways)} ways from {ways_pickle_path}")

        isects_pickle_path = os.path.join(dataset_dir, "all_intersections.pkl")
        with open(isects_pickle_path, "rb") as f:
            self.isect_gt = np.array(pickle.load(f))
        rospy.loginfo(
            f"Imported {self.isect_gt.shape[0]} intersections from {isects_pickle_path}"
        )

        selected_isects_pickle_path = os.path.join(
            dataset_dir, "selected_intersections.pkl"
        )
        with open(selected_isects_pickle_path, "rb") as f:
            self.selected_isect_gt = np.array(pickle.load(f))
        rospy.loginfo(
            f"Imported {self.selected_isect_gt.shape[0]} selected intersections from {selected_isects_pickle_path}"
        )


    def run_eval(self, road_graph: RoadGraph, threshold=50.0):
        isect_est = [n for n in road_graph.get_nodes() if type(n) == Intersection]
        isect_est = [n.get_center_point() for n in isect_est]
        isect_est = np.array(
            list(ros_numpy.numpify(i) for i in isect_est if i is not None)
        )
        if isect_est.shape[0] == 0:
            rospy.logwarn_throttle(20.0, "OSM eval: No intersections found in the road graph")
            return

        msg = IntersectionMetric()

        precision, recall, mean_dst, mean_dst_all = self.compute_stats(isect_est, self.selected_isect_gt, threshold)
        rospy.loginfo(
            f"Intersection stats across selected OSM intersections: Precision={precision:.2f}, Recall={recall:.2f}, Mean err (associated)={mean_dst:.2f}, Mean err (all)={mean_dst_all:.2f}"
        )

        msg.precision_sel = precision
        msg.recall_sel = recall
        msg.mean_dst_sel_all = mean_dst_all
        msg.mean_dst_sel_assoc = mean_dst

        precision, recall, mean_dst, mean_dst_all = self.compute_stats(isect_est, self.isect_gt, threshold)
        rospy.loginfo(
            f"Intersection stats across all OSM intersections: Precision={precision:.2f}, Recall={recall:.2f}, Mean err (associated)={mean_dst:.2f}, Mean err (all)={mean_dst_all:.2f}"
        )

        msg.precision_all = precision
        msg.recall_all = recall
        msg.mean_dst_all_all = mean_dst_all
        msg.mean_dst_all_assoc = mean_dst

        msg.header.stamp = rospy.Time.now()
        self.metric_pub.publish(msg)
        

    def compute_stats(
        self, isect_est: NDArray, isect_gt, assoc_threshold=50.0
    ):
        # build KD trees
        isect_gt_kdt = KDTree(isect_gt)

        isect_est_2d = isect_est[:, :2]  # project to 2d
        dst, gt_idxs = isect_gt_kdt.query(isect_est_2d)
        est_idxs = np.where(dst < assoc_threshold)[0]
        isect_est_3d = isect_est[est_idxs]
        gt_idxs = gt_idxs[dst < assoc_threshold]

        true_pos = len(est_idxs)
        false_pos = isect_est.shape[0] - true_pos
        false_neg = isect_gt.shape[0] - true_pos

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        mean_dst = np.mean(dst[est_idxs])
        mean_dst_all = np.mean(dst)
        self.plot_associations(isect_est_3d, isect_gt[gt_idxs])

        return (precision, recall, mean_dst, mean_dst_all)

    def plot_associations(
        self,
        isect_est,
        isect_gt,
    ):
        ma = MarkerArray()
        ma.markers = []
        for i, (est, gt) in enumerate(zip(isect_est, isect_gt)):
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = rospy.Time.now()
            m.ns = "associations"
            m.id = i
            m.type = m.ARROW
            m.pose.orientation.w = 1.0
            m.scale.x = 0.5
            m.scale.y = 0.1
            m.scale.z = 0.1
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 1.0
            m.points = []
            m.points.append(Point(gt[0], gt[1], 0.0))
            m.points.append(Point(est[0], est[1], est[2]))
            ma.markers.append(m)
        self.marker_pub.publish(ma)
