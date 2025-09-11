from abc import ABC, abstractmethod

from copy import deepcopy
from geometry_msgs.msg import Point
from hdl_graph_slam.msg._Keyframe_msg import Keyframe_msg
import rospy
import numpy as np
import ros_numpy
from typing import List, Union

from hdl_graph_slam.msg import KeyframeArray_msg
from visualization_msgs.msg import Marker

from .base_classes import SGLayer, SGNode


class Intersection(SGNode):
    def __init__(self, id):
        self.id = id
        self.keyframes = []
        self.reload_params()
        self.center: Point

    def reload_params(self):
        self.marker_color: List = rospy.get_param("~curb/road_graph/intersection_color")  # type: ignore
        self.marker_alpha: float = rospy.get_param("~curb/road_graph/road_graph_alpha")  # type: ignore
        self.marker_id_scale: float = rospy.get_param("~curb/road_graph/intersection_id_scale")  # type: ignore
        self.z_offset: float = rospy.get_param("~curb/scene_graph/road_graph_offset")  # type: ignore
        
    def add_keyframe(self, keyframe):
        self.keyframes.append(keyframe)

        # update center point
        pts = np.vstack([ros_numpy.numpify(kf.odom.position) for kf in self.keyframes])  # type: ignore
        centroid = np.mean(pts, axis=0)
        self.center = ros_numpy.msgify(Point, centroid)
        self.center.z += self.z_offset

    def get_center_point(self) -> Union[None, Point]:
        return self.center

    def get_id_marker(self) -> Union[None, Marker]:
        m = self.get_marker()
        if m is None:
            return None

        m.id = (self.id + 1) * 1000
        m.pose.position.z += rospy.get_param("~curb/road_graph/intersection_id_height")  # type: ignore
        m.type = m.TEXT_VIEW_FACING
        m.text = str(int(self.id))
        m.scale.z = self.marker_id_scale

        return m

    def get_marker(self) -> Union[Marker, None]:
        # scale = np.max(np.max(pts, axis=0) - np.min(pts, axis=0))
        scale: float = rospy.get_param("~curb/road_graph/intersection_scale")  # type: ignore
        # scale = 100.0

        m = Marker()
        m.action = m.ADD
        m.id = self.id
        m.header.frame_id = "world"
        m.ns = "road_graph"

        m.color.a = self.marker_alpha
        m.color.r = self.marker_color[0]
        m.color.g = self.marker_color[1]
        m.color.b = self.marker_color[2]

        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0

        pos = self.get_center_point()
        if pos is None:
            return None
        m.pose.position = pos

        m.type = m.SPHERE
        m.scale.x = scale
        m.scale.y = scale
        m.scale.z = scale

        return m


class Road(SGNode):
    def __init__(self, id, intersection1, intersection2):
        self.id = id
        self.int1 = intersection1
        self.int2 = intersection2
        self.keyframes: List[Keyframe_msg] = []
        self.reload_params()
        self.center: Point

    def reload_params(self):
        self.marker_color: List[float] = rospy.get_param("~curb/road_graph/road_color")  # type: ignore
        self.marker_alpha: float = rospy.get_param("~curb/road_graph/road_graph_alpha")  # type: ignore
        self.road_marker_scale: float = rospy.get_param("~curb/road_graph/road_scale")  # type: ignore
        self.z_offset: float = rospy.get_param("~curb/scene_graph/road_graph_offset")  # type: ignore
        
    def add_keyframe(self, keyframe):
        self.keyframes.append(keyframe)
        # self.keyframes.sort(key=lambda kf: kf.accum_distance)

        # update center
        center_idx = len(self.keyframes) // 2
        self.center = deepcopy(self.keyframes[center_idx].odom.position)
        self.center.z += self.z_offset

    def find_kf_pos(self, kf_id: int) -> Union[None, Point]:
        for kf in self.keyframes:
            if kf.id == kf_id:
                p = deepcopy(kf.odom.position)
                p.z += self.z_offset
                return p
        return None

    def get_center_point(self) -> Union[None, Point]:
        return self.center

    def get_marker(self):
        m = Marker()
        m.points = []

        m.action = m.ADD
        m.id = (self.id[0] + 1) * 10000 + (self.id[1])
        m.header.frame_id = "world"
        m.type = m.LINE_STRIP
        m.ns = "road_graph"

        m.color.a = self.marker_alpha
        m.color.r = self.marker_color[0]
        m.color.g = self.marker_color[1]
        m.color.b = self.marker_color[2]

        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0

        m.pose.position.z = self.z_offset

        for kf in self.keyframes:
            m.points.append(kf.odom.position)

        m.scale.x = self.road_marker_scale

        return m



class RoadGraph(SGLayer):
    """ Base class for building a road graph from agent trajectories.
    """
    def __init__(self):
        super().__init__()
        self.valid = False

        self.num_agents: int = rospy.get_param("~curb/num_agents")  # type: ignore

        self.keyframes_sub = rospy.Subscriber(
            "keyframes", KeyframeArray_msg, self.keyframes_callback, queue_size=1
        )

    def reload_params(self):
        for road in self.roads.values():
            road.reload_params()
        for intersection in self.intersections.values():
            intersection.reload_params()
        
    def reset(self):
        """Reset the road graph layer. if overwritten, make sure to call super().reset()"""
        self.keyframes_by_agent = [[] for _ in range(self.num_agents)]
        self.agent_trajectories = [[] for _ in range(self.num_agents)]

        self.intersections = dict()
        self.roads = dict()
        self.kf2node_map = dict()

        self.reload_params()

    @abstractmethod
    def compute_graph(self):
        """Overwrite this with code to compute the road graph. Will be
        automatically called when agent trajectories are updated and lock was
        acquired.
        should fill:
        self.intersections: dict of intersections
        self.roads: dict of roads
        self.kf2node_map: dict mapping keyframe ids to nodes
        """
        pass

    def get_nodes(self) -> List[Union[Intersection, Road]]:
        """retrieve a list of all nodes in the graph. Calling function should acquire our lock first."""
        if self.valid:
            # return all nodes as a list
            return list(self.roads.values()) + list(self.intersections.values())
        else:
            return []

    def find_kf_pos(self, kf_id: int) -> Union[None, Point]:
        """finds the point in the road graph corresponding to the given key
        frame id. If the keyframe is part of a road, return the point on the
        road. If the keyframe is part of an intersection, return the center of
        that intersection.
        """
        with self.lock:
            if kf_id not in self.kf2node_map.keys():
                return None
            node = self.kf2node_map[kf_id]
            if isinstance(node, Road):
                pos = node.find_kf_pos(kf_id)
            elif isinstance(node, Intersection):
                pos = node.get_center_point()
            else:
                pos = None

            assert (
                pos is not None
            ), f"pos for kf {kf_id} returned from node should never be None"

            return pos

    def keyframes_callback(self, keyframes: KeyframeArray_msg):
        # new keyframes
        self.reset()
        with self.lock:
            self.extract_agent_trajectories(keyframes)
            self.compute_graph()

    def extract_agent_trajectories(self, keyframes: KeyframeArray_msg):
        # seperate keyframes by agent id and sort by keyframe id to get
        # agent trajectories
        for agent_id in range(self.num_agents):
            if keyframes.keyframes:
                self.keyframes_by_agent[agent_id] = [
                    kf for kf in keyframes.keyframes if kf.agent_no == agent_id
                ]
                self.keyframes_by_agent[agent_id].sort(key=lambda kf: kf.accum_distance)

        # translate keyframes to SE(3) poses
        for agent_id in range(self.num_agents):
            for kf in self.keyframes_by_agent[agent_id]:
                # pose: 4x4 matrix
                pose = ros_numpy.numpify(kf.odom)
                self.agent_trajectories[agent_id].append(pose)
