import enum
from typing import Dict, List, Union

import cv2
from tf import TransformListener
import numpy as np
from numpy.typing import NDArray
from curb_scene_graph import (
    LandMarkNode,
    LandMarkLayer
)
import ros_numpy
import rospy
import os

from curb_projection import CameraModel
from hdl_graph_slam.msg import Keyframe_msg, KeyframeArray_msg
from std_msgs.msg import Bool

from threading import Lock

from .utils import find_closest_sample, read_stamps, StampedImg, draw_landmark


class KeyframeReprojector:
    def __init__(self, landmarklayer: LandMarkLayer):
        self.tl = TransformListener()

        models_dir = rospy.get_param("~models_dir")
        self.model = CameraModel(models_dir, "stereo/left")
        self.lock = Lock()
        self.keyframes: Union[None, List[Keyframe_msg]] = None
        self.landmarklayer = landmarklayer
        self.culling_range = 50.0
        self.keyframe_sub = rospy.Subscriber(
            "/optimized_keyframes", KeyframeArray_msg, self.keyframe_cb
        )
        self.cmd_sub = rospy.Subscriber(
            "/reproject_all_keyframes", Bool, self.reproject_all_kfs
        )


    def keyframe_cb(self, msg: KeyframeArray_msg):
        if msg.keyframes is None or len(msg.keyframes) == 0:
            return  # no keyframes

        with self.lock:
            # store keyframes msg
            self.keyframes = msg.keyframes

    def reproject_all_kfs(self, msg: Bool):
        with self.lock:
            if self.keyframes is None:
                return
            self.landmarklayer.graph_changed_callback(None)
            with self.landmarklayer.lock:
                landmarks = self.landmarklayer.get_nodes()
                self.reproject_landmarks(landmarks)

    def reproject_landmarks(self, landmarks: List[LandMarkNode]):
        assert self.keyframes is not None
        agent_nos = list(set([kf.agent_no for kf in self.keyframes]))
        datasets: Dict[int, str] = dict()
        timedeltas: Dict[int, float] = dict()
        imgs: Dict[int, List[StampedImg]] = dict()
        for agent_no in agent_nos:
            dataset: str = rospy.get_param(f"/robotcar_{agent_no}/dataset")  # type: ignore
            rospy.loginfo(f"dataset path for agent {agent_no}: {dataset}")
            datasets[agent_no] = os.path.join(dataset, "stereo/left")
            imgs[agent_no] = read_stamps(datasets[agent_no])
            # offset parameter to align keyframes with agent's oxford clock
            timedeltas[agent_no] = rospy.get_param(  # type: ignore
                f"/robotcar_{agent_no}/timediff_wall2oxford"
            )

            rospy.loginfo(
                f"waiting for transform from robotcar_{agent_no}/base_link to robotcar_{agent_no}/stereo_left"
            )
            self.tl.waitForTransform(
                f"robotcar_{agent_no}/stereo_left",
                f"robotcar_{agent_no}/base_link",
                rospy.Time(0),
                rospy.Duration(10),
            )
            t, r = self.tl.lookupTransform(
                f"robotcar_{agent_no}/stereo_left",
                f"robotcar_{agent_no}/base_link",
                rospy.Time(0),
            )
            tf_baselink2cam = self.tl.fromTranslationRotation(t, r)

        n = len(self.keyframes)
        for i, kf in enumerate(self.keyframes):
            if i % 50 == 0:
                rospy.loginfo(f"reprojecting keyframe {i+1}/{n}")
            self.reproject_into_keyframe(
                kf,
                landmarks,
                imgs[kf.agent_no],
                timedeltas[kf.agent_no],
                tf_baselink2cam,
            )

    def reproject_into_keyframe(
        self,
        kf: Keyframe_msg,
        landmarks: List[LandMarkNode],
        imgs: List[StampedImg],
        timediff: float,
        tf_baselink2cam: List[float],
    ):
        keyframe_stamp_wall = kf.header.stamp.to_sec()
        keyframe_stamp_oxf = keyframe_stamp_wall - timediff

        img_path = find_closest_sample(keyframe_stamp_oxf, imgs)
        assert os.path.exists(img_path), f"img {img_path} does not exist"
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.demosaicing(img, cv2.COLOR_BayerGR2RGB)
        img = self.model.undistort(img)

        keyframe_pos: NDArray = ros_numpy.numpify(kf.odom.position)  # type: ignore
        keyframe_tf: NDArray = ros_numpy.numpify(kf.odom)  # type: ignore
        world2kf = np.linalg.inv(keyframe_tf)

        # find landmarks within range
        close_landmarks: List[LandMarkNode] = []
        for lm in landmarks:
            lm_pos = lm.global_position
            dist = np.linalg.norm(keyframe_pos - lm_pos)
            if dist < self.culling_range:
                close_landmarks.append(lm)
        
        if len(close_landmarks) == 0:
            return

        overlay = np.zeros(img.shape, dtype=np.uint8)
        for lm in close_landmarks:
            # reproject
            assert lm.global_position is not None and lm.global_cloud is not None

            lm_pos = np.reshape(lm.global_position, (-1, 1))
            # transform to camera frame
            transform = tf_baselink2cam @ world2kf
            lm_pos = np.dot(transform, np.vstack([lm_pos, [1]]))[:3]
            lm_cloud = np.dot(
                transform,
                np.vstack([lm.global_cloud.T, np.ones(lm.global_cloud.shape[0])]),
            ).T[:, :3]

            center_uv, _, _ = self.model.project(lm_pos.reshape(-1,3).T, img.shape)
            pts_uv, depths, _ = self.model.project(lm_cloud.T, img.shape)
            if center_uv.shape[1] == 0 or pts_uv.shape[1] == 0:
                continue

            # add observation
            overlay = overlay + draw_landmark(
                img, center_uv, pts_uv, depths, lm.rgba[:3]
            )
        if overlay.sum() == 0:
            return  # no markers to show

        result = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
        result =  cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            f"/workspaces/collaborative-scene-graphs/imgdebug/reproj/repr_{kf.id:07}.png",
            result,
        )
