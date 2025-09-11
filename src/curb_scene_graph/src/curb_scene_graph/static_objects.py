import timeit
from sklearn.cluster import DBSCAN
from aabbtree import AABB

from curb_projection.msg._StaticObjectObs import StaticObjectObs
from curb_projection.msg._StaticObjectObsArray import StaticObjectObsArray
from curb_projection.utils import ColorGen
from geometry_msgs.msg import Point
import ros_numpy
import rospy
import ros_numpy.point_cloud2 as pcl2
import numpy as np
from numpy.typing import NDArray
from typing import List, Union
from tf import TransformListener
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool

from .base_classes import SGLayer, SGNode


class LandMarkObservation:
    """A single observation of a static object."""

    def __init__(self, static_object_msg: StaticObjectObs, tl: TransformListener):
        self.tl = tl

        self.kf_id = static_object_msg.keyframe_id
        self.stamp = static_object_msg.header.stamp
        self.local_position = static_object_msg.centroid
        self.instance_id = static_object_msg.instance_id
        self.observing_agent_id = static_object_msg.observing_agent_id
        self.class_name: str = static_object_msg.class_name.data
        self.certainty = static_object_msg.certainty
        self.local_cloud = pcl2.pointcloud2_to_xyz_array(static_object_msg.cloud)
        self.clip_feature = np.array(static_object_msg.clip_feature)

        assert (
            self.local_cloud is not None and self.local_cloud.shape[0] >= 2
        ), "No cloud available"

        self.global_position: Union[NDArray, None] = None
        self.global_cloud: Union[NDArray, None] = None
        self.aabb: Union[AABB, None] = None

        # when the observation is first, received, wait until the keyframe is available
        self.update_global_position(wait_time=4)

    def update_global_position(self, wait_time: int = 0):
        try:
            self.tl.waitForTransform(
                "world",
                f"keyframe/{self.kf_id}",
                rospy.Time(0),
                timeout=rospy.Duration(wait_time),
            )
            t, r = self.tl.lookupTransform(
                "world",
                f"keyframe/{self.kf_id}",
                rospy.Time(0),
            )
            # build homogeneous transformation matrix
            kf_in_world = self.tl.fromTranslationRotation(t, r)
        except Exception as e:
            rospy.logerr(f"Failed to lookup transform: {e}")
            self.global_position = None
            self.global_cloud = None
            return

        self.global_position = np.dot(
            kf_in_world,
            np.array(
                [self.local_position.x, self.local_position.y, self.local_position.z, 1]
            ),
        )[:3]

        self.global_cloud = np.dot(
            kf_in_world,
            np.hstack([self.local_cloud, np.ones((self.local_cloud.shape[0], 1))]).T,
        ).T[:, :3]

        minpt = np.min(self.global_cloud, axis=0)
        maxpt = np.max(self.global_cloud, axis=0)
        limits = np.vstack([minpt, maxpt]).T
        self.aabb = AABB(limits)

    def is_valid(self) -> bool:
        return (
            self.global_position is not None
            and self.global_cloud is not None
            and self.aabb is not None
        )

    def dist(self, other: "LandMarkObservation") -> float:
        if self.global_position is None or other.global_position is None:
            return np.inf

        return float(np.linalg.norm(self.global_position - other.global_position))

    def overlap_volume(self, other: "LandMarkObservation") -> float:
        if self.aabb is None or other.aabb is None:
            return -1.0
        if self.aabb.overlaps(other.aabb):
            return self.aabb.overlap_volume(other.aabb)
        return 0.0


class LandMarkNode(SGNode):
    """A static object node in the scene graph."""

    def __init__(self, observation: LandMarkObservation, tl: TransformListener):
        self.tl = tl
        self.class_name: str = observation.class_name

        self.reload_params()

        rgb = ColorGen().get_color(self.class_name)
        self.rgba = [rgb[0], rgb[1], rgb[2], self.alpha]

        self.observations: List[LandMarkObservation] = []
        self.global_position: Union[NDArray, None] = None
        self.global_cloud: Union[NDArray, None] = None
        self.aabb: Union[AABB, None] = None
        self.add_observation(observation)
        self.original_aabb = self.aabb

        self.scattered = (
            False  # a node is scattered if its observations are too far apart
        )
        self.shifted = (
            False  # a node is shifted if it moved too far from its original position
        )

        self.merged_to_other = False

    def reload_params(self):
        # params
        self.z_offset: float = rospy.get_param("~curb/scene_graph/landmarks_offset")  # type: ignore
        self.alpha: float = rospy.get_param("~curb/landmarks/objects_alpha")  # type: ignore
        self.min_obs: int = rospy.get_param("~curb/landmarks/min_obs_per_landmark/" + self.class_name)  # type: ignore
        # self.assoc_radius: float = rospy.get_param("~curb/landmarks/assoc_radius/" + self.class_name)  # type: ignore
        self.min_overlap_ratio: float = rospy.get_param("~curb/landmarks/overlap_ratio")  # type: ignore
        self.marker_scale_factor: float = rospy.get_param("~curb/landmarks/marker_scale_factor")  # type: ignore
        self.show_scores: bool = rospy.get_param("~curb/landmarks/show_scores")  # type: ignore

    def valid_association(
        self, other: Union[LandMarkObservation, "LandMarkNode"]
    ) -> bool:
        if other.class_name != self.class_name:
            return False
        valid = self.is_valid() and other.is_valid()
        if not valid:
            return False
        assert self.aabb is not None and other.aabb is not None
        overlap = self.overlap_volume(other)
        smaller_vol = min(self.aabb.volume, other.aabb.volume)
        overlap_ratio = overlap / smaller_vol

        return overlap_ratio >= self.min_overlap_ratio

    def get_kf_id(self) -> int:
        """return the id of the first observation"""
        return self.observations[0].kf_id

    def dist(self, other: LandMarkObservation) -> float:
        if self.global_position is None or other.global_position is None:
            return np.inf

        return float(np.linalg.norm(self.global_position - other.global_position))

    def overlap_volume(
        self, other: Union[LandMarkObservation, "LandMarkNode"]
    ) -> float:
        if self.aabb is None or other.aabb is None:
            return -1.0
        if self.aabb.overlaps(other.aabb):
            return self.aabb.overlap_volume(other.aabb)
        return 0.0

    def add_observation(self, landmark_obs: LandMarkObservation):
        assert self.class_name == landmark_obs.class_name, "Class mismatch"
        self.observations.append(landmark_obs)
        self.set_global_position_and_cloud()

    def retransform_observations(self):
        for obs in self.observations:
            obs.update_global_position()
        self.set_global_position_and_cloud()
        if not self.shifted and self.aabb is not None and self.original_aabb is not None:
            self.shifted = (
                not self.aabb.overlaps(self.original_aabb)
                or self.aabb.overlap_volume(self.original_aabb) < self.min_overlap_ratio
            )
            if self.shifted:
                self.original_aabb = self.aabb

    def set_global_position_and_cloud(self):
        # check if all global positions are available
        assert len(self.observations) > 0, "No observations available"
        if any(obs.global_position is None for obs in self.observations):
            self.global_position = None
            self.global_cloud = None
            self.aabb = None
            return

        # average global position
        pos_array = np.vstack([obs.global_position for obs in self.observations])  # type: ignore
        self.global_position = np.mean(pos_array, axis=0)

        # merge all clouds
        self.global_cloud = np.vstack([obs.global_cloud for obs in self.observations])  # type: ignore

        # merge all aabbs
        self.aabb = self.observations[0].aabb
        for obs in self.observations[1:]:
            self.aabb = AABB.merge(self.aabb, obs.aabb)

        # node is scattered if one of the obs does not have enough overlap
        self.scattered = False
        for obs in self.observations:
            if not self.valid_association(obs):
                self.scattered = True
                break

    def avg_certainty(self) -> float:
        return float(np.mean([obs.certainty for obs in self.observations]))
        
    def get_center_point(self) -> Point:
        pt = Point(*self.global_position)
        pt.z += self.z_offset
        return pt

    def is_valid(self) -> bool:
        return (
            self.global_position is not None
            and self.global_cloud is not None
            and self.aabb is not None
        )
    
    def enough_obs(self) -> bool:
        return len(self.observations) >= self.min_obs

    def split(self) -> List["LandMarkNode"]:
        # if the observations are too far apart, split the landmark into N single-observation landmarks
        # otherwise return an empty list
        if not self.scattered:
            return []
        self.scattered = False
        outlier_obs = [
            obs
            for obs in self.observations
            if not obs.is_valid() or not self.valid_association(obs)
        ]
        inlier_obs = [obs for obs in self.observations if obs not in outlier_obs]
        if len(inlier_obs) == 0:
            # no inliers left, take one of the outlier observations for the current landmark
            self.add_observation(outlier_obs[0])
            if len(outlier_obs) > 1:
                # create new landmarks for the remaining outliers
                return [LandMarkNode(obs, self.tl) for obs in outlier_obs[1:]]
            else:
                # only one observation, no need to split
                return []
        self.observations = inlier_obs
        self.set_global_position_and_cloud()
        new_landmarks = [LandMarkNode(obs, self.tl) for obs in outlier_obs]
        return new_landmarks

    def merge(self, other: "LandMarkNode"):
        self.observations.extend(other.observations)
        self.set_global_position_and_cloud()
        other.observations = []
        other.merged_to_other = True

    def get_marker(self) -> MarkerArray:
        ma = MarkerArray()
        ma.markers = []

        box_marker = Marker()
        box_marker.action = box_marker.ADD
        # create unique id from first obs keyframe id and instance id
        box_marker.id = self.observations[0].kf_id * 100 + self.observations[0].instance_id
        box_marker.ns = f"landmark_{self.class_name}"
        box_marker.header.frame_id = "world"
        box_marker.type = box_marker.CUBE

        box_marker.color.r = self.rgba[0]
        box_marker.color.g = self.rgba[1]
        box_marker.color.b = self.rgba[2]
        box_marker.color.a = self.rgba[3]

        assert self.global_cloud is not None, "No global cloud available"

        # compute bounding box from cloud
        # TODO: compute bounding box from AABB
        min_xyz = np.min(self.global_cloud, axis=0)
        max_xyz = np.max(self.global_cloud, axis=0)
        box_marker.pose.position.x = (min_xyz[0] + max_xyz[0]) / 2
        box_marker.pose.position.y = (min_xyz[1] + max_xyz[1]) / 2
        box_marker.pose.position.z = (min_xyz[2] + max_xyz[2]) / 2 + self.z_offset
        box_marker.scale.x = (max_xyz[0] - min_xyz[0]) * self.marker_scale_factor
        box_marker.scale.y = (max_xyz[1] - min_xyz[1]) * self.marker_scale_factor
        box_marker.scale.z = (max_xyz[2] - min_xyz[2]) * self.marker_scale_factor
        box_marker.pose.orientation.w = 1

        ma.markers.append(box_marker)
        if self.show_scores:
            score_marker = Marker()
            score_marker.action = score_marker.ADD
            score_marker.id = self.observations[0].kf_id * 200 + self.observations[0].instance_id
            score_marker.ns = f"landmark_scores_{self.class_name}"
            score_marker.header.frame_id = "world"
            score_marker.type = score_marker.TEXT_VIEW_FACING
            score_marker.text = f"[{self.class_name}: {len(self.observations)} obs; {self.avg_certainty():.3f}]"
            score_marker.pose.position.x = box_marker.pose.position.x
            score_marker.pose.position.y = box_marker.pose.position.y
            score_marker.pose.position.z = box_marker.pose.position.z - box_marker.scale.z
            score_marker.scale.z = box_marker.scale.z / 2
            score_marker.color.r = 0.5
            score_marker.color.g = 0.5
            score_marker.color.b = 0.5
            score_marker.color.a = 1.0

            ma.markers.append(score_marker)

        return ma


class LandMarkLayer(SGLayer):
    def __init__(self, tl: TransformListener):
        self.objects_sub = rospy.Subscriber(
            "static_objects",
            StaticObjectObsArray,
            self.static_obs_callback,
            queue_size=1,
        )

        self.graph_changed_sub = rospy.Subscriber(
            "/graph_changed", Bool, self.graph_changed_callback
        )

        # initialize the mutex from the parent class
        super().__init__()

        self.new_observations = 0

        self.tl = tl
        self.reset()

    def reset(self):
        with self.lock:
            self.landmarks: List[LandMarkNode] = []
            self.observations: List[StaticObjectObs] = []
            self.reload_params()
    
    def reload_params(self):
        self.landmark_targets: str = rospy.get_param("~curb/landmarks/landmark_targets")  # type: ignore
        for lm in self.landmarks:
            lm.reload_params()

    def get_nodes(self) -> List[LandMarkNode]:
        return [lm for lm in self.landmarks if lm.is_valid() and lm.enough_obs()]

    # TODO: code for creating, joining and splitting landmarks when new observations
    # come in or pose graph changes

    def static_obs_callback(self, static_objects_msg: StaticObjectObsArray):
        if (
            not static_objects_msg.observations
            or len(static_objects_msg.observations) == 0
        ):
            return  # help the type checker figure out that observations is not None

        for static_object_msg in static_objects_msg.observations:
            assert (
                static_object_msg.class_name.data in self.landmark_targets
            ), "Unknown class"
            lm_node = LandMarkObservation(static_object_msg, self.tl)
            with self.lock:
                found_association = False
                # check if observation can be associated with existing landmark
                for lm in self.landmarks:
                    if lm.valid_association(lm_node):
                        lm.add_observation(lm_node)
                        found_association = True
                        break
                if not found_association:
                    # create new landmark
                    self.landmarks.append(LandMarkNode(lm_node, self.tl))

    def graph_changed_callback(self, msg: Bool):
        # first wait until the new poses arrived
        rospy.sleep(3)
        rospy.loginfo("Landmark layer: graph changed, retransforming observations")
        # now trigger retransformation of all observations
        with self.lock:
            for lm in self.landmarks:
                lm.retransform_observations()
            self.update_associations()

    def update_associations(self):
        """Split and merge landmarks based on associations"""
        # try to split landmarks that moved
        for lm in self.landmarks:
            if not lm.scattered:
                continue
            # split the landmark if it is too far apart. If the landmark is
            # consistent, it will return a list with only the original landmark
            split_nodes = lm.split()
            self.landmarks.extend(split_nodes)
            if len(split_nodes) > 0:
                rospy.loginfo(
                    f"Splitting {lm.class_name} landmark into {len(split_nodes) + 1}"
                )

        assert all(len(lm.observations) > 0 for lm in self.landmarks)

        # try to find new associations and merge landmarks
        for i, lm in enumerate(self.landmarks[:-1]):
            if lm.merged_to_other or not lm.shifted:
                continue  # skip already merged landmarks and landmarks that did not move
            for other_lm in self.landmarks[i + 1 :]:
                if other_lm.merged_to_other:
                    continue  # skip already merged landmarks
                if lm != other_lm and lm.valid_association(other_lm):
                    lm.merge(other_lm)
                    other_lm.merged_to_other = True
                    rospy.loginfo(
                        f"Merging {lm.class_name} landmarks with {len(lm.observations)} observations"
                    )
        self.landmarks = [lm for lm in self.landmarks if not lm.merged_to_other]

        # make sure we have no empty landmarks
        assert all(len(lm.observations) > 0 for lm in self.landmarks)
