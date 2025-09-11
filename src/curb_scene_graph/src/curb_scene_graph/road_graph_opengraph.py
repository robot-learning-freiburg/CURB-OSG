from copy import deepcopy
from curb_scene_graph.road_graph import Intersection, Road, RoadGraph
from geometry_msgs.msg import Point
from hdl_graph_slam.msg._Keyframe_msg import Keyframe_msg
import rospy
import numpy as np
from sklearn.cluster import DBSCAN


class RoadGraphOG(RoadGraph):
    """
    Road graph is extracted by the method proposed in the paper:
    "OpenGraph: Open-Vocabulary Hierarchical 3D Graph Representation in Large-Scale
    Outdoor Environments" by Deng et al (2024)
    http://arxiv.org/abs/2403.09412
    """

    def __init__(self):
        super().__init__()

        self.disfluency_thresh: float = rospy.get_param("~curb/road_graph/disfluency_thresh")  # type: ignore
        self.r_dis: float = rospy.get_param("~curb/road_graph/r_dis")  # type: ignore

        dbscan_eps: float = rospy.get_param("~curb/road_graph/dbscan_eps")  # type: ignore
        dbscan_min_samples: int = rospy.get_param("~curb/road_graph/dbscan_min_samples")  # type: ignore

        self.dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)

        # state will be set in reset:
        self.reset()

    def reset(self):
        with self.lock:
            super().reset()
            self.disfluencies = [[] for _ in range(self.num_agents)]

    def compute_graph(self):
        """Compute the road graph using the OpenGraph method. Assumes that the
        lock is already acquired."""

        self.valid = self.compute_disfluencies()
        if not self.valid:
            rospy.logwarn(
                "Road graph: Local disfluency computation failed, no road graph extracted!"
            )
            return

        self.valid = self.build_graph()
        if not self.valid:
            rospy.logwarn("Road graph: graph building failed, no road graph extracted!")

    def compute_disfluencies(self):
        window: int = rospy.get_param("~curb/road_graph/neighbor_search_window")  # type: ignore
        for agent_id in range(self.num_agents):
            for n, p_n in enumerate(self.agent_trajectories[agent_id]):
                # v_n is the set of neighborhood vectors of this trajectory point
                v_n = []
                # only the 2d translation is considered
                p_n_2d = p_n[0:2, 3]

                # consider points along the trajectory within a window. This is
                # to avoid high disfluency if a trajectory moves near itself (i.e. two-way road)
                for m in range(
                    max(0, n - window),
                    min(n + window, len(self.agent_trajectories[agent_id])),
                ):
                    p_m = self.agent_trajectories[agent_id][m]
                    p_m_2d = p_m[0:2, 3]
                    v_diff = p_n_2d - p_m_2d
                    dist = np.linalg.norm(v_diff)
                    # only consider neighbors within this radius, don't consider self
                    if dist != 0.0 and dist <= self.r_dis:
                        v_n.append(v_diff)
                if len(v_n) <= 1:
                    # no neighbors!
                    return False

                # calculate pairwise angles' difference to 0 or pi
                angle_diffs = []
                for i in range(len(v_n) - 1):
                    for j in range(i + 1, len(v_n)):
                        cos_angle = np.dot(v_n[i], v_n[j]) / (
                            np.linalg.norm(v_n[i]) * np.linalg.norm(v_n[j])
                        )
                        angle = np.arccos(cos_angle)
                        # difference between angle and 0 or pi
                        angle_diff = min(abs(angle), abs(angle - np.pi))
                        angle_diffs.append(angle_diff)

                # finally, the disfluency measure for this trajectory point
                lambda_n = np.mean(angle_diffs)
                self.disfluencies[agent_id].append(lambda_n)

        return True

    def build_graph(self):
        if len(self.agent_trajectories) == 0 or all(
            len(traj) == 0 for traj in self.agent_trajectories
        ):
            return False

        # get Nx2 array of 2d trajectory points of _all_ agents
        points_2d = np.vstack(
            [p[:2, 3] for traj in self.agent_trajectories for p in traj]
        )
        # get the disfluencies as a flattened numpy array
        dis_np = np.concatenate(
            [np.array(agent_dis) for agent_dis in self.disfluencies]
        )

        # get keyframes in the same shape to retain the mapping
        keyframes = [kf for agent_kfs in self.keyframes_by_agent for kf in agent_kfs]

        # points with high disfluency: intersection candidates
        candidate_pts = points_2d[dis_np > self.disfluency_thresh]

        if len(candidate_pts) == 0:
            return False

        point_idxs = np.where(dis_np > self.disfluency_thresh)[0]

        # run DBSCAN clustering to find intersections
        self.dbscan.fit(candidate_pts)
        clusters = self.dbscan.labels_

        # generate the graph by marching along the clusters, generating
        # intersection nodes and road edges along the way
        prev_idx = point_idxs[0]
        prev_intersection = -1
        prev_agent = -1
        for this_intersection, this_idx in zip(clusters, point_idxs):
            if this_intersection == -1:
                # DBSCAN marks noisy samples with -1
                continue

            # generate intersection node if it does not exist
            if this_intersection not in self.intersections.keys():
                self.intersections[this_intersection] = Intersection(this_intersection)

            # add all keyframes belonging to this intersection
            self.intersections[this_intersection].add_keyframe(keyframes[this_idx])

            # store mapping to find node from kf id
            self.kf2node_map[keyframes[this_idx].id] = self.intersections[
                this_intersection
            ]

            this_agent = keyframes[this_idx].agent_no
            if prev_agent != this_agent or prev_intersection == -1:
                # trajectories do not belong together, do not add edge
                prev_agent = this_agent
                prev_intersection = this_intersection
                continue

            if prev_intersection != this_intersection:
                # entering a new cluster, add edge from the previous cluster if
                # it does not exist
                edge_id = (prev_intersection, this_intersection, this_agent)
                if edge_id not in self.roads.keys():
                    self.roads[edge_id] = Road(
                        edge_id,
                        self.intersections[prev_intersection],
                        self.intersections[this_intersection],
                    )

                for i in range(prev_idx, this_idx + 1):
                    self.roads[edge_id].add_keyframe(keyframes[i])
                    if i in range(prev_idx + 1, this_idx):
                        # store mapping to find node from kf id
                        self.kf2node_map[keyframes[i].id] = self.roads[edge_id]

                prev_intersection = this_intersection

            prev_idx = this_idx
        return True
