"""
This script is used to automatically launch and collect metrics with different agent counts
"""

from typing import List, Tuple, Union

from attr import field
from hdl_graph_slam.msg._KeyframeArray_msg import KeyframeArray_msg
import rosbag
import rospy
from curb_metrics.msg import ATEMetric, IntersectionMetric, KITTIMetric
import subprocess
import os
import time
import datetime
import csv
from itertools import combinations, product
from std_msgs.msg import Bool, String

LAUNCHFILE = "/workspaces/collaborative-scene-graphs/src/curb_launchers/launch/curb_osg.launch"
LOG_DIR = "/workspaces/collaborative-scene-graphs/metrics/"
RESUME_DIR = None
# RESUME_DIR = "/workspaces/collaborative-scene-graphs/metrics/2025-02-28T17:19"
AGENTS = [0, 1, 2]
N_AGENTS = [1,2,3]
DRY_RUN = False
SEMSEG_BASELINE = False  # if True, use semantic segmentation MaskCLIP baseline, else OG pipeline
ONLY_AGENT_1 = False # if True, only evaluate with agent 1
LM_DETECTION = False
VARY_DYN_OBJS = False  # if True, also evaluate with and without dynamic objects

INTERSECTIONS_EVAL = True
if INTERSECTIONS_EVAL:
    SEMSEG_BASELINE = False
    ONLY_AGENT_1 = False
    LM_DETECTION = False
    VARY_DYN_OBJS = False
    N_AGENTS = [2, 3]  # [1,2,3]
    


class RunLog:
    def __init__(
        self,
        session_dir: str,
        which_agents: Tuple[int, ...],
        remove_dyn_objs: bool,
        args: List[str],
    ):
        self.which_agents = which_agents
        self.agent_comb = "".join([str(i) for i in self.which_agents])
        self.remove_dyn_objs = remove_dyn_objs
        self.args = " ".join(args)

        self.ate_metric_msgs: List[ATEMetric] = []
        self.intersections_metric_msgs: List[IntersectionMetric] = []
        self.kitti_metric_msgs: List[KITTIMetric] = []

        self.last_change_time = time.time()

        self.skip = False

        self.init_logdir(session_dir)
        if self.skip:
            return

        rospy.loginfo(f"Run log initiated with " + self.__str__())

    def add_metric(self, metric: Union[ATEMetric, IntersectionMetric, KITTIMetric]):
        if isinstance(metric, ATEMetric):
            if not self.metric_equal_to_last(metric):
                self.last_change_time = time.time()
            self.ate_metric_msgs.append(metric)
        elif isinstance(metric, IntersectionMetric):
            if not self.metric_equal_to_last(metric):
                self.last_change_time = time.time()
            self.intersections_metric_msgs.append(metric)
        elif isinstance(metric, KITTIMetric):
            if not self.metric_equal_to_last(metric):
                self.last_change_time = time.time()
            self.kitti_metric_msgs.append(metric)
        else:
            raise ValueError("Unknown metric type")

        self.write_log(metric)

    def metric_equal_to_last(self, metric):
        if isinstance(metric, ATEMetric):
            if len(self.ate_metric_msgs) == 0:
                return False
            return self.ate_metric_msgs[-1].mean_ate == metric.mean_ate
        elif isinstance(metric, IntersectionMetric):
            if len(self.intersections_metric_msgs) == 0:
                return False
            metric2 = self.intersections_metric_msgs[-1]
            return (
                metric.precision_all == metric2.precision_all
                and metric.recall_all == metric2.recall_all
                and metric.mean_dst_all_all == metric2.mean_dst_all_all
                and metric.mean_dst_all_assoc == metric2.mean_dst_all_assoc
                and metric.precision_sel == metric2.precision_sel
                and metric.recall_sel == metric2.recall_sel
                and metric.mean_dst_sel_all == metric2.mean_dst_sel_all
                and metric.mean_dst_sel_assoc == metric2.mean_dst_sel_assoc
            )
        elif isinstance(metric, KITTIMetric):
            if len(self.kitti_metric_msgs) == 0:
                return False
            metric2 = self.kitti_metric_msgs[-1]
            return (
                metric.mean_transl_err == metric2.mean_transl_err
                and metric.mean_rot_err == metric2.mean_rot_err
            )
        else:
            raise ValueError(f"Unknown metric type: {metric}")

    def last_metric_str(self):
        str = ""
        if len(self.ate_metric_msgs) > 0:
            str += f"ATE: {self.ate_metric_msgs[-1].mean_ate:.3f}, "
        if len(self.intersections_metric_msgs) > 0:
            str += f"Intersections: P={self.intersections_metric_msgs[-1].precision_all:.3f}, R={self.intersections_metric_msgs[-1].recall_all:.3f}, "
        if len(self.kitti_metric_msgs) > 0:
            str += f"KITTI: transl_err={self.kitti_metric_msgs[-1].mean_transl_err:.3f}, rot_err={self.kitti_metric_msgs[-1].mean_rot_err:.3f}"
        if str == "":
            str = "No metrics received yet"
        return str

    def time_since_last_change(self):
        return time.time() - self.last_change_time

    def __str__(self):
        str = f"agents={self.agent_comb}, remove_dyn_objs={self.remove_dyn_objs}, last_change_time {self.time_since_last_change()} seconds ago"
        return str

    def collect_semseg_predictions(self):
        if not 1 in self.which_agents or self.remove_dyn_objs:
            # only if agent 1 is running or dyn objs are removed we can not reproject
            return

        rospy.loginfo("Collecting semantic segmentation predictions")
        pub = rospy.Publisher("/run_gt_reprojection", String, queue_size=1, latch=True)
        pub.publish(self.logdir)
        
        try:
            rospy.wait_for_message("/gt_reprojection_done", Bool, timeout=600)
        except rospy.exceptions.ROSException:
            rospy.logwarn("GT reprojection took longer than 10 minutes! aborting")
        else:
            rospy.loginfo(f"Semantic segmentation predictions collected in {self.logdir}")

        pub.unregister()
        
    def init_logdir(self, sessiondir: str):
        # create dir with ISO date for this run
        self.logdir = os.path.join(
            sessiondir,
            f"agents-{self.agent_comb}_remove_dyn_objs-{self.remove_dyn_objs}",
        )
        try:
            os.makedirs(self.logdir)
        except FileExistsError:
            rospy.loginfo(f"Run {self.logdir} already exists, skipping")
            self.skip = True
            return

        # create and open log file for appending
        self.logfile = open(os.path.join(self.logdir, "log.txt"), "a+")
        self.rosbag_logfile = open(os.path.join(self.logdir, "rosbag.log"), "a+")

        # initiate CSV files for each metric
        self.ate_csv = open(os.path.join(self.logdir, "ate.csv"), "a+")
        self.ate_writer = csv.DictWriter(
            self.ate_csv, fieldnames=["timestamp", "mean_ate", "std_ate"]
        )
        self.ate_writer.writeheader()

        self.intersections_csv = open(
            os.path.join(self.logdir, "intersections.csv"), "a+"
        )
        self.intersections_writer = csv.DictWriter(
            self.intersections_csv,
            fieldnames=[
                "timestamp",
                "precision_all",
                "recall_all",
                "mean_dst_all_all",
                "mean_dst_all_assoc",
                "precision_sel",
                "recall_sel",
                "mean_dst_sel_all",
                "mean_dst_sel_assoc",
            ],
        )
        self.intersections_writer.writeheader()

        self.kitti_csv = open(os.path.join(self.logdir, "kitti.csv"), "a+")
        self.kitti_writer = csv.DictWriter(
            self.kitti_csv,
            fieldnames=[
                "timestamp",
                "mean_transl_err",
                "std_transl_err",
                "mean_rot_err",
                "std_rot_err",
            ],
        )
        self.kitti_writer.writeheader()

        self.init_rosbag(os.path.join(self.logdir, "pose_graphs.bag"))

        rospy.loginfo(f"Logging metrics to {self.logdir}")

    def init_rosbag(self, target_file):
        self.log = os.path.join(target_file, ".log")
        self.rosbag_process = subprocess.Popen(
            ["rosbag", "record", "-O", target_file, "/optimized_keyframes", "/tf", "/tf_static"],
            stdout=self.rosbag_logfile,
            stderr=self.rosbag_logfile
        )


    def close(self):
        self.ate_csv.close()
        self.intersections_csv.close()
        self.kitti_csv.close()
        self.logfile.close()
        self.rosbag_logfile.close()
        self.rosbag_process.terminate()
        self.rosbag_process.wait(timeout=10)
        if self.rosbag_process.returncode is None:
            rospy.logwarn("rosbag process did not terminate, killing")
            self.rosbag_process.kill()
        

    def write_log(self, metric: Union[ATEMetric, IntersectionMetric, KITTIMetric]):
        if isinstance(metric, ATEMetric):
            self.ate_writer.writerow(
                {
                    "timestamp": metric.header.stamp.to_sec(),
                    "mean_ate": metric.mean_ate,
                    "std_ate": metric.std_ate,
                }
            )
        elif isinstance(metric, IntersectionMetric):
            self.intersections_writer.writerow(
                {
                    "timestamp": metric.header.stamp.to_sec(),
                    "precision_all": metric.precision_all,
                    "recall_all": metric.recall_all,
                    "mean_dst_all_all": metric.mean_dst_all_all,
                    "mean_dst_all_assoc": metric.mean_dst_all_assoc,
                    "precision_sel": metric.precision_sel,
                    "recall_sel": metric.recall_sel,
                    "mean_dst_sel_all": metric.mean_dst_sel_all,
                    "mean_dst_sel_assoc": metric.mean_dst_sel_assoc,
                }
            )
        elif isinstance(metric, KITTIMetric):
            self.kitti_writer.writerow(
                {
                    "timestamp": metric.header.stamp.to_sec(),
                    "mean_transl_err": metric.mean_transl_err,
                    "std_transl_err": metric.std_transl_err,
                    "mean_rot_err": metric.mean_rot_err,
                    "std_rot_err": metric.std_rot_err,
                }
            )
        else:
            raise ValueError("Unknown metric type")


class AutoEvalNode:
    def __init__(self):
        rospy.init_node("auto_eval_node")
        rospy.set_param("/use_sim_time", False)

        self.ate_sub = rospy.Subscriber("/metric_ate", ATEMetric, self.metric_callback)
        self.kitti_sub = rospy.Subscriber(
            "/metric_kitti", KITTIMetric, self.metric_callback
        )
        self.intersections_sub = rospy.Subscriber(
            "/metric_intersections",
            IntersectionMetric,
            self.metric_callback,
        )

        self.keyframes_sub = rospy.Subscriber(
            "/optimized_keyframes", KeyframeArray_msg, self.keyframes_callback
        )
        self.keyframes_cts = [0]

        self.runlog = None

        rospy.loginfo("Auto eval node started")

        self.create_session_dir()
        self.eval()

    def create_session_dir(self):
        # create logdir if it does not exist
        # create dir with ISO date for this eval session
        if not os.path.exists(LOG_DIR) and not DRY_RUN:
            os.makedirs(LOG_DIR)
        if RESUME_DIR:
            rospy.loginfo(f"Resuming from {RESUME_DIR}")
            self.session_dir = RESUME_DIR
        else:
            self.session_dir = os.path.join(
                LOG_DIR,
                f"{datetime.datetime.now().isoformat(timespec='minutes')}",
            )
            rospy.loginfo(f"Logging to {self.session_dir}")
            if not DRY_RUN:
                os.makedirs(self.session_dir)

    def eval(self):
        for n_agents in N_AGENTS:
            if n_agents == 1:
                clock_multiplier = 1.0
            elif n_agents == 2:
                clock_multiplier = 0.6
            else:
                clock_multiplier = 0.4
            
            if INTERSECTIONS_EVAL:
                clock_multiplier = 1.4  # gt gps is fast

            agent_combinations = list(combinations(AGENTS, n_agents))
            if ONLY_AGENT_1:
                # only evaluate combinations that include agent 1
                agent_combinations = [comb for comb in agent_combinations if 1 in comb]
            if VARY_DYN_OBJS:
                param_combinations = list(product(agent_combinations, [True, False]))
            else:
                param_combinations = [(comb, False) for comb in agent_combinations]
            for which_agents, remove_dyn_objs in param_combinations:
                # if len(which_agents) == 1 and which_agents[0] == 0:
                #     # skip single agent with agent 0 only
                #     continue
                if rospy.is_shutdown():
                    return
                args = self.generate_args(
                    which_agents, remove_dyn_objs, clock_multiplier
                )

                if DRY_RUN:
                    rospy.loginfo(f"Would run {which_agents} with args: {' '.join(args)}")
                else:
                    self.runlog = RunLog(
                        self.session_dir, which_agents, remove_dyn_objs, args
                    )
                    if self.runlog.skip:
                        continue
                    rospy.loginfo("-----------------------------")
                    rospy.loginfo(f"starting new run with args: {' '.join(args)}")
                    self.launch_run(args, clock_multiplier)
                    time.sleep(20)  # time to store logs and let roscore clear
                    self.runlog.close()
                    self.runlog = None
                    rospy.loginfo("run finished")

    def generate_args(
        self,
        which_agents: Tuple[int, ...],
        remove_dyn_objs: bool,
        clock_multiplier: float,
    ) -> List[str]:
        args = [
            "roslaunch",
            LAUNCHFILE,
            f"remove_dynamic_objects:={str(remove_dyn_objs).lower()}",
            f"detect_landmarks:={str(LM_DETECTION and 1 in which_agents).lower()}",
            f"clock_multiplier:={clock_multiplier}",
            f"semseg_baseline:={str(SEMSEG_BASELINE).lower()}",
        ]
        agent_no = 0
        for i in AGENTS:
            if i in which_agents:
                args.append(f"run_agent_{i}:=true")
                args.append(f"agent_no_{i}:={agent_no}")
                # first agent serves as fixed global frame; agents 1,... must be shifted
                # no shift for intersections eval
                args.append(f"shift_agent_{i}:={str(agent_no!=0 and not INTERSECTIONS_EVAL).lower()}")
                agent_no += 1
            else:
                args.append(f"run_agent_{i}:=false")
        return args

    def launch_run(self, args: List[str], clock_multiplier: float):
        assert self.runlog is not None
        # initiate subprocess
        process = subprocess.Popen(
            args,
            stdout=self.runlog.logfile,
            stderr=self.runlog.logfile,
        )

        elapsed = 0
        semseg_trial_complete = False
        kill = False
        while not rospy.is_shutdown():
            # try exporting semseg predictions early to test
            if elapsed > 60 and not semseg_trial_complete:
                self.runlog.collect_semseg_predictions()
                semseg_trial_complete = True
                
            if self.runlog.time_since_last_change() > 120:
                rospy.loginfo(
                    f"No change in metrics for {self.runlog.time_since_last_change()} seconds"
                )
                kill = True
            if elapsed > 2100 / clock_multiplier:
                rospy.loginfo(f"Run took longer than {2100 / clock_multiplier} seconds")
                kill = True
                
            if kill:
                self.runlog.collect_semseg_predictions()
                rospy.loginfo("Killing process")
                process.terminate()
                # wait for process to finish
                try:
                    process.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    rospy.logwarn("Process did not terminate, killing")
                    process.kill()
                if not process.returncode:
                    rospy.logwarn("Process did not terminate (no return code), killing")
                    process.kill()
                break

            rospy.loginfo_throttle(180.0, self.runlog.last_metric_str())

            time.sleep(1)
            elapsed += 1

    def metric_callback(self, metric):
        if self.runlog is not None and not self.runlog.skip:
            self.runlog.add_metric(metric)
        else:
            rospy.logwarn("Received metric but no runlog initiated")

    def keyframes_callback(self, keyframes_array: KeyframeArray_msg):
        assert keyframes_array.keyframes is not None
        self.keyframes_cts.append(len(keyframes_array.keyframes))
        # TODO: check if all agents have finished


if __name__ == "__main__":
    AutoEvalNode()
    rospy.loginfo("Auto eval node finished")
    exit(0)
