# evaluation of keyframe poses against SE(2) ground-truth odometry
# processes bag file recordings of keyframe pose graphs

from argparse import ArgumentError
import copy
import re
from typing import List, Tuple

from click import Argument
from cv2 import transform
from tqdm import tqdm
from numpy.typing import NDArray
import rospy
from hdl_graph_slam.msg import Keyframe_msg, KeyframeArray_msg
import rosbag
import os
import csv
import sys
import numpy as np
from tf import Transformer, TransformListener
from geometry_msgs.msg import TransformStamped, Transform, Pose
from tf import transformations
from ros_numpy import numpify, msgify
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion
from absolute_trajectory_error import compute_ate
from odometry_metrics import eval_odometry_metrics, OdomMetricSample
import matplotlib.pyplot as plt


def plot_trajectory(trajectory, color, label):
    trajectory = np.array([t[:2, 3] for t in trajectory]).T
    plt.plot(trajectory[0, :], trajectory[1, :], color, label=label)


class SE2AgentEval:
    def __init__(self):

        rospy.init_node("se2_agent_eval")
        self.marker_pub = rospy.Publisher("/ate_dbg", Marker)

        self.transformer = Transformer(
            interpolate=True, cache_time=rospy.Duration(10000)
        )

        self.fromtranslationrotation = TransformListener().fromTranslationRotation

        for i in range(3):
            delta_t = float(rospy.get_param(f"/robotcar_{i}/timediff_wall2oxford"))
            dataset = str(rospy.get_param(f"/robotcar_{i}/dataset"))
            print(f"loading dataset {dataset} for agent {i}")
            gt_file = os.path.join(dataset, "gt/radar_odometry.csv")
            trajectory = self.read_gt(gt_file, delta_t, i)
            # plot_trajectory(trajectory, f"C{i}", f"agent{i}_gt")

        # plt.savefig("gt_trajectories.png")
        # sys.exit(0)

    def read_gt(self, gt_file, delta_t, agent_no):
        trajectory = []
        with open(gt_file) as f:
            gt_dict = csv.DictReader(f)
            t_numpy = None
            t = TransformStamped()
            t.header.frame_id = f"robotcar_{agent_no}/se2_gt"
            # t.header.frame_id = "world"
            t.child_frame_id = f"robotcar_{agent_no}/base_link"
            for line in tqdm(gt_dict):
                # tgt is where we were, src is where we are (very confusing)
                src_time_wall = float(line["source_timestamp"]) / 1e6 + delta_t
                tgt_time_wall = float(line["destination_timestamp"]) / 1e6 + delta_t
                if tgt_time_wall < 0.0:
                    continue

                if t_numpy is None:
                    # set first transform to identity
                    t_numpy = transformations.identity_matrix()
                    # t_numpy = initials[agent_no]
                    t.header.stamp = rospy.Time.from_sec(tgt_time_wall)
                    t.transform = msgify(Transform, t_numpy)
                    self.transformer.setTransform(t)
                    print(f"agent{agent_no} setting first transform at {tgt_time_wall}")

                # we are now at src, first build the transform from previous tgt stamp
                yaw = float(line["yaw"])
                x = float(line["x"])
                y = float(line["y"])
                t_rel = transformations.compose_matrix(
                    angles=(0.0, 0.0, yaw), translate=(x, y, 0.0)
                )

                # chain transformations to get current pose in first
                t_numpy = t_numpy @ t_rel
                t.header.stamp = rospy.Time.from_sec(src_time_wall)

                t.transform = msgify(Transform, t_numpy)
                self.transformer.setTransform(t)
                trajectory.append(t_numpy)
        return trajectory

    def eval_posegraph(self, keyframes: List[Keyframe_msg], agent_no_translator: dict) -> Tuple[Tuple[float, float], OdomMetricSample]:
        agent_trajectories = [[], [], []]
        first_stamps = [None, None, None]
        first_transforms: List[NDArray] = [None, None, None]
        gt_trajectories = [[], [], []]

        gt_transformer = Transformer(interpolate=True, cache_time=rospy.Duration(10000))
        ates = []
        agent_nos = []

        keyframes_2d = []

        keyframes.sort(key=lambda k: k.id)
        # initials = [None, None, None]
        for kf in keyframes:
            n = kf.agent_no
            q = kf.odom.orientation
            t = kf.odom.position
            p = self.fromtranslationrotation([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])
            agent_trajectories[n].append(p)
            if first_stamps[n] is None:
                first_stamps[n] = kf.header.stamp
                t, r = self.transformer.lookupTransform(
                    "world", f"robotcar_{n}/gt/base_link", first_stamps[n]
                )
                first_transforms[n] = self.fromtranslationrotation(t, r)

            # get transform in frame of first keyframe
            odom_frame = f"robotcar_{agent_no_translator[n]}/se2_gt"
            base_frame = f"robotcar_{agent_no_translator[n]}/base_link"
            t, r = self.transformer.lookupTransform(
                odom_frame, base_frame, first_stamps[n]
            )
            t_gtodom2first = self.fromtranslationrotation(t, r)

            t, r = self.transformer.lookupTransform(
                odom_frame, base_frame, kf.header.stamp
            )

            t_gtodom2current = self.fromtranslationrotation(t, r)

            t_gtfirst2current = np.linalg.inv(t_gtodom2first) @ t_gtodom2current

            t_gtworld2current = first_transforms[n] @ t_gtfirst2current

            gt_trajectories[n].append(t_gtworld2current)

            t = TransformStamped()
            t.header.frame_id = "world"
            t.child_frame_id = f"robotcar_{n}/gt/base_link"
            t.transform = msgify(Transform, t_gtworld2current)
            t.header.stamp = kf.header.stamp
            gt_transformer.setTransform(t)

            kf_2d = copy.deepcopy(kf)
            kf_2d.odom.position.z = 0.0
            q = kf_2d.odom.orientation
            q = [q.x, q.y, q.z, q.w]
            rpy = transformations.euler_from_quaternion(q)
            q = transformations.quaternion_from_euler(0.0, 0.0, rpy[2])
            kf_2d.odom.orientation = msgify(Quaternion, q)
            keyframes_2d.append(kf_2d)

            # ate_se2 = np.linalg.norm(t_gtworld2current[:2, 3] - p[:2, 3])

            # ates.append(ate_se2)
            agent_nos.append(n)
        
        
        ate_result = compute_ate(keyframes_2d, gt_transformer)
        kitti_result = eval_odometry_metrics(gt_transformer, keyframes_2d)


        i = 0
        for a_traj, gt_traj in zip(agent_trajectories, gt_trajectories):
            self.plot_trajectory(i, gt_traj, 1.0, "gt")
            self.plot_trajectory(i, a_traj, 0.5, "agent")
            i += 1

        return ate_result, kitti_result

    def plot_trajectory(self, i, gt_traj, alpha, ns):
        gt_marker = Marker()
        gt_marker.header.frame_id = "world"
        gt_marker.action = gt_marker.ADD
        gt_marker.type = gt_marker.LINE_STRIP
        gt_marker.color.a = alpha
        gt_marker.color.r = 1.0 if i == 0 else 0.0
        gt_marker.color.g = 1.0 if i == 1 else 0.0
        gt_marker.color.b = 1.0 if i == 2 else 0.0
        gt_marker.scale.x = 10.0
        gt_marker.scale.y = 10.0
        gt_marker.scale.z = 10.0
        gt_marker.pose.orientation.w = 1.0
        gt_marker.ns = ns
        gt_marker.id = i
        gt_marker.points = []
        for gt_t in gt_traj:
            p = Point()
            p.x = gt_t[0, 3]
            p.y = gt_t[1, 3]
            p.z = 0.0
            gt_marker.points.append(p)

        self.marker_pub.publish(gt_marker)

    def agent_no_translator(self, rundir):
        match = re.match(r".*agents-(\d+)_remove_dyn_objs-(True|False).?", rundir)
        if match is None:
            print(f"could not parse run directory {rundir}")
            return
        agents = match.group(1)

        agent_no_translator = {}
        if len(agents) == 1:
            agent_no_translator[0] = int(agents[0])
        elif len(agents) == 2:
            agent_no_translator[0] = int(agents[0])
            agent_no_translator[1] = int(agents[1])
        elif len(agents) == 3:
            agent_no_translator[0] = int(agents[0])
            agent_no_translator[1] = int(agents[1])
            agent_no_translator[2] = int(agents[2])
        else:
            raise Exception(f"unexpected number of agents: {agents}")

        return agent_no_translator

    def read_bag_tf(self, bag):
        for topic, msg, t in bag.read_messages(topics=["/tf", "/tf_static"]):
            for transform in msg.transforms:
                if "gt/base_link" in transform.child_frame_id:
                    self.transformer.setTransform(transform)

    def eval_run(self, rundir):
        print(f"evaluating run in {rundir}")
        agent_no_translator = self.agent_no_translator(rundir)

        keyframe_poses = []
        bag = rosbag.Bag(os.path.join(rundir, "pose_graphs.bag"))
        kfs = None
        self.ate_csv = open(os.path.join(rundir, "se2_ate.csv"), "w")
        self.ate_writer = csv.DictWriter(
            self.ate_csv, fieldnames=["timestamp", "mean_ate", "std_ate"]
        )
        self.ate_writer.writeheader()

        self.kitti_csv = open(os.path.join(rundir, "se2_kitti.csv"), "w")
        self.kitti_writer = csv.DictWriter(
            self.kitti_csv, fieldnames=["timestamp","mean_transl_err","std_transl_err","mean_rot_err","std_rot_err" ]
        )
        self.kitti_writer.writeheader()

        self.read_bag_tf(bag)
        i = 0
        for topic, msg, t in tqdm(bag.read_messages(topics=["/optimized_keyframes"])):
            if i % (100*len(agent_no_translator.keys())) != 0:
                i += 1
                continue
            if rospy.is_shutdown():
                break
            msg: KeyframeArray_msg
            msg.header.stamp
            (mate, std_ate), kitti = self.eval_posegraph(msg.keyframes, agent_no_translator)
            self.ate_writer.writerow(
                {
                    "timestamp": msg.header.stamp.to_sec(),
                    "mean_ate": mate,
                    "std_ate": std_ate,
                }
            )
            self.kitti_writer.writerow(
                {
                    "timestamp": msg.header.stamp.to_sec(),
                    "mean_transl_err": kitti.mean_transl_err,
                    "std_transl_err": kitti.std_transl_err,
                    "mean_rot_err": kitti.mean_rot_err,
                    "std_rot_err": kitti.std_rot_err,
                }
            )
            # if i % 2000 == 0:
            #     print(
            #         f"timestamp: {msg.header.stamp.to_sec()}, mean_ate: {mate}, mean_transl_err: {kitti.mean_transl_err}, mean_rot_err: {kitti.mean_rot_err}"
            #     )

        bag.close()

        return keyframe_poses


def main(rundir):
    assert os.path.isdir(rundir)

    subdirs = [d for d in os.listdir(rundir) if os.path.isdir(os.path.join(rundir, d))]
    if len(subdirs) == 0:
        print(f"no subdirs found in {rundir}")
        return
    print(f"found {len(subdirs)} subdirs in {rundir}")

    for d in subdirs:
        if rospy.is_shutdown():
            break
        evaluator = SE2AgentEval()
        evaluator.eval_run(os.path.join(rundir, d))
    # rospy.spin()


if __name__ == "__main__":
    main(sys.argv[1])
