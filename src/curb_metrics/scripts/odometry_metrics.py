import time
import os
from os import path
from typing import List, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import ros_numpy
import rospy
from tf import TransformListener, transformations, Transformer

from curb_metrics.msg import KITTIMetric as KITTIMetricMsg

from hdl_graph_slam.msg import Keyframe_msg, KeyframeArray_msg

# maximum duration of a run in seconds
MAX_RUN_DURATION = 3600  # 60 min
LENGTHS: List[int] = [100, 200, 300, 400, 500, 600, 700, 800]

class OdomMetricSample:
    def __init__(self, transl_errors: List[float], rot_errors: List[float]):
        self.transl_errors = transl_errors
        self.rot_errors = rot_errors

        self.mean_rot_err = float(np.mean(self.rot_errors))
        self.mean_transl_err = float(np.mean(self.transl_errors))
        self.std_rot_err = float(np.std(self.rot_errors))
        self.std_transl_err = float(np.std(self.transl_errors))

    def get_msg(self) -> KITTIMetricMsg:
        msg = KITTIMetricMsg()
        msg.header.stamp = rospy.Time.now()
        msg.mean_rot_err = self.mean_rot_err
        msg.mean_transl_err = self.mean_transl_err
        msg.std_rot_err = self.std_rot_err
        msg.std_transl_err = self.std_transl_err
        return msg

    def __add__(self, other: "OdomMetricSample") -> "OdomMetricSample":
        return OdomMetricSample(
            self.transl_errors + other.transl_errors, self.rot_errors + other.rot_errors
        )

    def __str__(self) -> str:
        return f"Odometry metrics across all agents: e_rot [deg/m] {self.mean_rot_err:.5f}+-{self.std_rot_err:.5f} e_trans [m/m] {self.mean_transl_err:.5f}+-{self.std_transl_err:.5f}"

def get_keyframes_odom(keyframes_array: List[Keyframe_msg]) -> Tuple[List[List[NDArray]], List[List[rospy.Time]]]:
    if len(keyframes_array) == 0:
        return [], []

    # list of sequential keyframe transforms for each agent, in world frame
    keyframes_global = [[], [], []]
    stamps = [[], [], []]
    first_keyframe_ids = [0, 1000000, 2000000]
    last_keyframe_ids = [0, 0, 0]

    # translate all keyframe msgs to transform matrixes and order them by
    # agent and id
    kf: Keyframe_msg
    for kf in keyframes_array:
        t = kf.odom.position
        q = kf.odom.orientation
        r = transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        kf_transform_global = transformations.compose_matrix(translate=(t.x, t.y, t.z), angles=r)
        keyframes_global[kf.agent_no].append(kf_transform_global)
        stamps[kf.agent_no].append(kf.header.stamp)

        # map keyframe id back from 0, 1e6, 2e6 space to 0...
        kf_id_normalized = kf.id - first_keyframe_ids[kf.agent_no]

        # make sure the next keyframe follows the last immediately
        assert (
            kf_id_normalized == 0
            or last_keyframe_ids[kf.agent_no] == kf_id_normalized - 1
        ), f"odom metric node: keyframes {last_keyframe_ids[kf.agent_no]} and {kf_id_normalized} of agent {kf.agent_no} out of order"

        last_keyframe_ids[kf.agent_no] = kf_id_normalized

    # transform each keyframe tf into the frame of the first keyframe
    keyframes_in_first = []
    for kf_seq in keyframes_global:
        if len(kf_seq) < 2:
            break

        first_keyframe_inv = np.linalg.inv(kf_seq[0])

        # first keyframe in its own frame is identity
        keyframes_in_first.append([np.eye(4, 4)])

        for kf in kf_seq[1:]:
            this_keyframe_in_first = first_keyframe_inv @ kf
            keyframes_in_first[-1].append(this_keyframe_in_first)
    
    return keyframes_in_first, stamps

def get_gt_poses(transformer, agent_id, stamps) -> List[NDArray]:
    # first get the ground truth poses at each time stamp, relative to first
    poses_gt: List[np.ndarray] = []
    for stamp in stamps:
        trans_gt, rot_gt = transformer.lookupTransformFull(
            f"robotcar_{agent_id}/gt/base_link",
            stamps[0],
            f"robotcar_{agent_id}/gt/base_link",
            stamp,
            "world",
        )

        pose_gt = transformations.compose_matrix(
            angles=transformations.euler_from_quaternion(rot_gt), translate=trans_gt
        )

        poses_gt.append(pose_gt)
    return poses_gt


def compute_trajectory_distances(poses: List[np.ndarray]) -> List[float]:
    dists = [0.0]
    for i in range(1, len(poses)):
        delta_trans = poses[i][:3, 3] - poses[i - 1][:3, 3]
        dists.append(dists[-1] + float(np.linalg.norm(delta_trans)))

    return dists

def compute_kitti_errors(pose_estimates, poses_gt) -> Tuple[List[float], List[float]]:
    rot_errors = []
    trans_errors = []

    dists = compute_trajectory_distances(poses_gt)

    # average over all start frames and lengths
    for first_frame_idx in range(len(poses_gt)):
        for length in LENGTHS:
            # find last frame at distance 'length' from first frame
            last_frame_idx = None
            for i in range(first_frame_idx, len(dists)):
                if dists[i] > dists[first_frame_idx] + length:
                    last_frame_idx = i
                    break
            if last_frame_idx is None:
                # sequence is shorter than this 'length'
                continue

            # compute error for this pair of frames
            pose_delta_gt = (
                np.linalg.inv(poses_gt[first_frame_idx]) @ poses_gt[last_frame_idx]
            )
            pose_delta_estimate = (
                np.linalg.inv(pose_estimates[first_frame_idx])
                @ pose_estimates[last_frame_idx]
            )
            pose_error = np.linalg.inv(pose_delta_estimate) @ pose_delta_gt

            # https://en.wikipedia.org/wiki/Rotation_matrix#Determining_the_angle
            E_rot = np.arccos((np.trace(pose_error[:3, :3]) - 1) / 2) / 3.14159 * 180  # deg/m

            E_trans = np.linalg.norm(pose_error[:3, 3]) * 100.0  # % (m/m)

            # todo: export more information, KITTI exports error by length and speed
            rot_errors.append(E_rot / length)
            trans_errors.append(E_trans / length)
    return rot_errors,trans_errors


def eval_agent(
    transformer: Union[TransformListener, Transformer],
    pose_estimates: List[np.ndarray],
    stamps: List[rospy.Time],
    agent_id: int,
) -> Union[OdomMetricSample, None]:
    """compute the translation and rotation error metric for a single
    agent's trajectory as average across different lengths
    """

    poses_gt = get_gt_poses(transformer, agent_id, stamps)

    # make sure all sequences have the same length
    assert (
        len(poses_gt) == len(pose_estimates) == len(stamps)
    ), "Odom metric node: Unexpected missmatch in sequence lengths"


    # if agent_id == 1:
    #     self.plot_poses(poses_gt, f"keyframe/{agent_id * 1000000}")
    #     # self.plot_poses(poses_gt, f'world')

    # self.write_poses(poses_gt, pose_estimates, agent_id)

    rot_errors, trans_errors = compute_kitti_errors(pose_estimates, poses_gt)

    if len(rot_errors) > 0:
        return OdomMetricSample(trans_errors, rot_errors)
    else:
        return None

def eval_all_agents(
    transformer: Union[TransformListener, Transformer],
    keyframe_sequences: List[List[np.ndarray]],
    stamp_sequences: List[List[rospy.Time]],
) -> OdomMetricSample:
    """compute the translation and rotation error metric for all agents"""
    odom_sample = OdomMetricSample([], [])

    for agent_id, kf_seq in enumerate(keyframe_sequences):
        this_odom_sample = eval_agent(
            transformer, kf_seq, stamp_sequences[agent_id], agent_id
        )
        if this_odom_sample is None:
            continue
        odom_sample += this_odom_sample

    return odom_sample

def eval_odometry_metrics(
    transformer: Union[TransformListener, Transformer], keyframes_array: List[Keyframe_msg]
) -> OdomMetricSample:

    keyframes_in_first, stamps = get_keyframes_odom(keyframes_array)
    odom_sample = eval_all_agents(transformer, keyframes_in_first, stamps)
    return odom_sample

class OdomMetricsNode:
    def __init__(self):

        self.keyframes_sub = rospy.Subscriber(
            "/optimized_keyframes", KeyframeArray_msg, self.keyframes_callback
        )

        self.metric_pub = rospy.Publisher("/metric_kitti", KITTIMetricMsg, queue_size=10)

        self.tf_listener = TransformListener(
            cache_time=rospy.Duration(MAX_RUN_DURATION), interpolate=True
        )
        self.start_time = rospy.Time.now()

        self.output_target_dir = path.join(
            f"/workspaces/collaborative-scene-graphs/src/curb_metrics/devkit/cpp/results/",
            str(int(time.time())),
            "data",
        )

        if not os.path.exists(self.output_target_dir):
            os.makedirs(self.output_target_dir)

        self.gt_target_dir = path.join(
            f"/workspaces/collaborative-scene-graphs/src/curb_metrics/devkit/cpp/poses/"
        )

        assert os.path.exists(self.gt_target_dir)

        print(f"Writing pose data to {self.gt_target_dir} and {self.output_target_dir}")


        rospy.loginfo("Odometry metrics node started")

    def keyframes_callback(self, keyframes_array: KeyframeArray_msg) -> None:

        assert keyframes_array.keyframes is not None
        odom_sample = eval_odometry_metrics(self.tf_listener, keyframes_array.keyframes)

        if len(odom_sample.transl_errors) == 0:
            # no errors reported because all sequences were shorter
            # than the shortest len
            return

        rospy.loginfo(str(odom_sample))

        self.metric_pub.publish(odom_sample.get_msg())

    def write_poses(self, poses_gt, pose_estimates, agent_no):

        outfile_gt = open(path.join(self.gt_target_dir, f"{agent_no}.txt"), "w")
        outfile_estimates = open(
            path.join(self.output_target_dir, f"{agent_no}.txt"), "w"
        )

        for poses, outfile in [
            (poses_gt, outfile_gt),
            (pose_estimates, outfile_estimates),
        ]:
            for p in poses:
                line = " ".join(str(x) for x in p[:3, :].flatten())
                outfile.write(line + "\n")

        outfile_gt.close()
        outfile_estimates.close()

    def plot_poses(self, poses: List[np.ndarray], frame: str):
        from geometry_msgs.msg import PoseArray, Pose

        pose_pub = rospy.Publisher(
            "/metric_pose_debug", PoseArray, latch=True, queue_size=10
        )
        m = PoseArray()
        m.poses = []
        m.header.frame_id = frame
        m.header.stamp = rospy.Time(0)
        for p in poses:
            p_msg = ros_numpy.msgify(Pose, p)
            m.poses.append(p_msg)
        pose_pub.publish(m)



if __name__ == "__main__":

    rospy.init_node("odom_metrics_node")
    rmse_node = OdomMetricsNode()
    rospy.spin()
