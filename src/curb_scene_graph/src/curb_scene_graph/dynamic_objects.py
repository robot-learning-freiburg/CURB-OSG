""" dynamic object layer of the scene graph.
"""

import rospy
import numpy as np
import ros_numpy
import sensor_msgs.point_cloud2 as pcl2
from typing import List, Tuple, Union
from tf import TransformListener, LookupException  # type: ignore

from curb_projection.msg import TrackedObjectObs, TrackedObjectObsArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped, Quaternion, Point

from .base_classes import SGLayer, SGNode

def cam_string_to_int(cam_str: str) -> int:
    if 'stereo' in cam_str:
        return 0
    elif 'left' in cam_str:
        return 1
    elif 'right' in cam_str:
        return 2
    elif 'rear' in cam_str:
        return 3
    else:
        rospy.logerr(f"Unknown camera string: {cam_str}")
        return -1


class DynamicObjectsLayer(SGLayer):
    def __init__(self, tl: TransformListener):
        super().__init__()
        self.tracked_obj_sub = rospy.Subscriber(
            "masa_observations",
            TrackedObjectObsArray,
            self.tracked_obj_callback,
            queue_size=16,
        )
        self.tl = tl

        # dict to map instance ids to TrackedObject
        self.tracked_objects: dict[Tuple[int, int, str], TrackedObject] = dict()

        self.our_instance_ids_cnt = 1000000
        # stores keys of instances which have been split up
        self.map_old2newkey = dict()

    def get_nodes(self):
        return list(self.tracked_objects.values())

    def tracked_obj_callback(self, observations_msg: TrackedObjectObsArray):
        if not observations_msg.observations or len(observations_msg.observations) == 0:
            return

        with self.lock:
            for obs in observations_msg.observations:
                obs: TrackedObjectObs

                # agent, instance and camera ids form a unique key for each tracked obj
                key = (obs.observing_agent_id, obs.instance_id, obs.camera.data)

                # check if we've seen this MASA instance before
                orig_instance_id_tracked = key in self.tracked_objects.keys()
                # check if we've split it up before and there is a newer key
                newer_split_found = key in self.map_old2newkey.keys()

                # handle all possible associations:
                # 1. no known association -> create new object
                if not orig_instance_id_tracked:
                    self.tracked_objects[key] = TrackedObject(self.tl)
                    self.tracked_objects[key].add_observation(obs)

                # 2. object tracked under MASA instance id, and no newer split found
                # -> try to match with original object
                if orig_instance_id_tracked and not newer_split_found:
                    # 2.1 objects match -> merge the two
                    if self.tracked_objects[key].is_same_instance(obs):
                        self.tracked_objects[key].add_observation(obs)
                    # 2.2 association failed -> split up
                    else:
                        self.split_instance(old_key=key, observation=obs)

                # 3. MASA track has been split before -> check if newest object matches
                if newer_split_found:
                    new_key = self.map_old2newkey[key]
                    # 3.1 objects match -> merge the two
                    if self.tracked_objects[new_key].is_same_instance(obs):
                        self.tracked_objects[new_key].add_observation(obs)
                    # 3.2 association failed -> split up
                    else:
                        self.split_instance(old_key=key, observation=obs)

    def split_instance(self, old_key, observation):
        """Split the tracked object. A new key will be created and referenced with the old key in self.map_old2newkey"""
        newkey = (old_key[0], self.our_instance_ids_cnt, old_key[2])
        self.our_instance_ids_cnt += 1
        self.map_old2newkey[old_key] = newkey
        self.tracked_objects[newkey] = TrackedObject(self.tl)
        self.tracked_objects[newkey].add_observation(observation)
        self.tracked_objects[newkey].instance_id = newkey[1]
    
    def reload_params(self):
        for obj in self.tracked_objects.values():
            obj.reload_params()


class TrackedObject(SGNode):
    def __init__(self, tl: TransformListener):
        """Represents a tracked object as a sequence of observations. Points is
        the point cloud of the object as Nx3 numpy array.
        """
        self.centroids: List[PointStamped] = []
        self.avg_clip_feature = None
        self.color = list(np.random.rand(3))
        self.instance_id = -1
        self.pointclouds = []
        self.observing_agent: int
        self.keyframe_ids = []
        
        self.reload_params()

        self.tl = tl

        # settings
        self.distance_thresh = 30
        self.static_object_radius = 8

    def reload_params(self):
        self.z_offset_parking: float = rospy.get_param("~curb/scene_graph/dynamic_objects_offset_parking")  # type: ignore
        self.z_offset_moving: float = rospy.get_param("~curb/scene_graph/dynamic_objects_offset_moving")  # type: ignore
        self.scale_parking: float = rospy.get_param("~curb/dynamic_objects/scale_parking")  # type: ignore
        self.scale_moving: float = rospy.get_param("~curb/dynamic_objects/scale_moving")  # type: ignore
        self.rgba_parking: List[float] = rospy.get_param("~curb/dynamic_objects/rgba_parking")  # type: ignore
        self.rgba_moving: List[float] = rospy.get_param("~curb/dynamic_objects/rgba_moving")  # type: ignore

    def get_center_point(self) -> Union[None, Point]:
        assert len(self.centroids) > 0
        center_idx = len(self.centroids) // 2
        pt = self.centroids[center_idx]
        try:
            pt.header.stamp = rospy.Time(0)
            pt = self.tl.transformPoint("world", pt).point
            if self.is_static():
                pt.z += self.z_offset_parking
            else:
                pt.z += self.z_offset_moving
            return pt
        except LookupException:
            return None

    def get_kf_id(self) -> int:
        assert len(self.keyframe_ids) > 0
        center_idx = len(self.keyframe_ids) // 2
        return self.keyframe_ids[center_idx]

    def add_observation(self, obs_msg: TrackedObjectObs):
        clip_feat = np.array(obs_msg.clip_feature)
        if self.avg_clip_feature is None:
            self.avg_clip_feature = clip_feat
        else:
            # online average formula
            n = len(self.centroids)
            self.avg_clip_feature += (clip_feat - self.avg_clip_feature) / n

        self.centroids.append(obs_msg.centroid)
        self.pointclouds.append(obs_msg.cloud)
        self.observing_agent = obs_msg.observing_agent_id
        self.keyframe_ids.append(obs_msg.keyframe_id)
        self.instance_id = obs_msg.instance_id
        self.camera = obs_msg.camera.data

    def is_same_instance(self, obs: TrackedObjectObs) -> bool:
        """Checks if an observation should be associated with this tracked object."""
        if len(self.centroids) == 0:
            # empty obs -> always accept association
            rospy.logwarn(
                "dynamic objects node: try to associate with empty observation"
            )
            return True
        pt_a: np.ndarray = ros_numpy.numpify(obs.centroid.point)  # type: ignore
        pt_b: np.ndarray = ros_numpy.numpify(self.centroids[-1].point)  # type: ignore
        dst = float(np.linalg.norm(pt_a - pt_b))

        return dst < self.distance_thresh

    def get_centroids_global(self):
        centroids_global = []
        for pt in self.centroids:
            try:
                pt.header.stamp = rospy.Time(0)
                pt_global = self.tl.transformPoint("world", pt).point
                centroids_global.append(pt_global)
            except LookupException:
                rospy.logdebug("could not find frame " + pt.header.frame_id)

        return centroids_global

    def is_static(self) -> bool:
        centroids_np: List[np.ndarray] = [
            ros_numpy.numpify(pt) for pt in self.get_centroids_global()  # type: ignore
        ]
        if len(centroids_np) == 0:
            return False

        if len(centroids_np) == 1:
            return True
        max_dist = 0.0
        for c in centroids_np[1:]:
            dist = float(np.linalg.norm(centroids_np[0] - c))
            if dist > max_dist:
                max_dist = dist

        return max_dist < self.static_object_radius

    def get_pcl2_msg(self) -> pcl2.PointCloud2:
        raise NotImplementedError()
        # todo

    def get_marker(self) -> Union[Marker, None]:
        m = Marker()
        m.color.a = 1.0
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = "world"
        m.action = Marker.ADD
        m.id = (self.observing_agent * 1000000) + self.instance_id * 1000 + cam_string_to_int(
            self.camera
        )
        m.pose.orientation = Quaternion(0, 0, 0, 1)

        m.points = self.get_centroids_global()
        am_static = self.is_static()

        m.ns = "tracked_objects_static" if am_static else "tracked_objects_dynamic"

        if len(m.points) == 0:
            rospy.logdebug(
                f"MASA node couldn't visualize object {self.instance_id}, no transforms found"
            )
            return None
        elif am_static:
            m.type = Marker.SPHERE
            m.scale.x = .3 * self.scale_parking
            m.scale.y = .3 * self.scale_parking
            m.scale.z = .3 * self.scale_parking
            m.pose.position = m.points[0]
            m.points.clear()
            rgba = self.rgba_parking
            m.pose.position.z += self.z_offset_parking
        else:
            # dynamic, create line marker
            m.type = Marker.LINE_STRIP
            m.scale.x = 0.3 * self.scale_moving
            rgba = self.rgba_moving
            m.pose.position.z += self.z_offset_moving

        m.color.r = rgba[0]
        m.color.g = rgba[1]
        m.color.b = rgba[2]
        m.color.a = rgba[3]

        return m
