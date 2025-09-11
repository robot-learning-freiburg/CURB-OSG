""" reproject landmarks for the 100 gt images exactly at their timestamp and
export predictions for evaluation
"""

from tabnanny import check
from typing import Dict, List, Optional, Tuple, Union
import rospy
import ros_numpy
import cv2
import os
import numpy as np
import json
from tf import TransformListener
from curb_scene_graph.static_objects import LandMarkLayer, LandMarkNode, NDArray
from curb_projection import CameraModel
from hdl_graph_slam.msg import KeyframeArray_msg, Keyframe_msg
from .utils import draw_landmark, draw_mask
import datetime
import pickle
from geometry_msgs.msg import PoseStamped, Pose, PoseArray
from std_msgs.msg import String, Bool
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from threading import Lock


class TimetravelReprojector:
    def __init__(self, landmarklayer: LandMarkLayer):
        self.valid = True
        self.tl = TransformListener(cache_time=rospy.Duration(7200), interpolate=True)

        models_dir = rospy.get_param("~models_dir")
        self.model = CameraModel(models_dir, "stereo/left")

        self.landmarklayer = landmarklayer

        self.default_target = (
            "/workspaces/collaborative-scene-graphs/src/curb_metrics/data/semseg_pred"
        )
        self.default_target = os.path.join(
            self.default_target, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )

        sequence = "2019-01-14-14-15-12-radar-oxford-10k"
        sequences: List[Tuple[int, str]] = []

        for n in [0, 1, 2]:
            try:
                if not rospy.get_param(f"/robotcar_{n}/running"):
                    continue
                sequences.append((n, rospy.get_param(f"/robotcar_{n}/dataset")))  # type: ignore
            except KeyError:
                pass

        try:
            agent_no = [n for n, seq in sequences if sequence in seq][0]
        except IndexError:
            rospy.logwarn(
                f"GT Eval Sequence {sequence} not found in any of the robotcar datasets"
            )
            self.valid = False
            return

        self.agent_no = agent_no

        dataset = rospy.get_param(f"/robotcar_{agent_no}/dataset")

        self.timediff_wall2oxf: float = rospy.get_param(  # type: ignore
            f"/robotcar_{agent_no}/timediff_wall2oxford"
        )

        gt_dataset = "/workspaces/collaborative-scene-graphs/src/curb_metrics/data/semseg_gt/robotcar-labeling-v1.0.json"
        self.gt_imgs = "/workspaces/collaborative-scene-graphs/src/curb_metrics/data/semseg_gt/test"
        with open(gt_dataset, "r") as f:
            gt_json = json.load(f)
        gt_stamps_oxf = [sample["name"] for sample in gt_json["dataset"]["samples"]]
        gt_stamps_oxf = [int(stamp.split(".")[0]) for stamp in gt_stamps_oxf]
        self.gt_stamps_oxf = sorted(gt_stamps_oxf)

        self.gt_stamps_wall = [
            t / 1e6 + self.timediff_wall2oxf for t in self.gt_stamps_oxf
        ]

        cats_json = gt_json["dataset"]["task_attributes"]["categories"]

        self.category2id = {c["name"]: c["id"] for c in cats_json}

        self.tl.waitForTransform(
            f"robotcar_{agent_no}/stereo_left",
            f"robotcar_{agent_no}/base_link",
            rospy.Time(0),
            rospy.Duration(40),
        )

        t, r = self.tl.lookupTransform(
            f"robotcar_{agent_no}/stereo_left",
            f"robotcar_{agent_no}/base_link",
            rospy.Time(0),
        )

        self.tf_baselink2cam = self.tl.fromTranslationRotation(t, r)

        self.global_map: Optional[NDArray] = None

        # self.map_points_sub = rospy.Subscriber(
        #     "/map_server/map_points", PointCloud2, self.global_map_callback
        # )
        self.keyframe_sub = rospy.Subscriber(
            "/optimized_keyframes", KeyframeArray_msg, self.keyframe_callback
        )

        self.done_pub = rospy.Publisher("/gt_reprojection_done", Bool, queue_size=1)

        self.debug_pose_pub = rospy.Publisher(
            "/gt_frame_pose", PoseArray, queue_size=100
        )
        self.debug_marker_pub = rospy.Publisher(
            "/gt_frame_pose_labels", MarkerArray, queue_size=100
        )

        # self.reproj_timer = rospy.Timer(rospy.Duration(60), self.run_reprojection)
        self.reproj_sub = rospy.Subscriber(
            "/run_gt_reprojection", String, self.repro_callback
        )

        self.cam_poses = []
        self.cam_pose_labels = []
        self.annotations = []

        self.keyframes: List[Keyframe_msg] = []
        self.lock = Lock()

        rospy.loginfo("Timetravel reprojection initialized")

    def keyframe_callback(self, kfs_msg: KeyframeArray_msg):
        if not self.valid or kfs_msg.keyframes is None or len(kfs_msg.keyframes) < 10:
            return
        with self.lock:
            self.keyframes = [
                kf for kf in kfs_msg.keyframes if kf.agent_no == self.agent_no
            ]

    def global_map_callback(self, msg: PointCloud2):
        if not msg.fields:
            return
        msg.fields.sort(key=lambda x: x.offset)
        self.global_map = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
        self.global_map_resolution = rospy.get_param("/map_server/map_cloud_resolution")

    def repro_callback(self, target_dir: String):
        self.run_reprojection(target_dir)
        self.done_pub.publish(Bool(True))

    def run_reprojection(self, target_dir: Optional[String] = None):
        if len(self.keyframes) < 2:
            rospy.logwarn("Not enough keyframes to reproject")
            return

        # if self.global_map is None:
        #     rospy.logwarn("No global map available")
        #     return

        kfs = self.keyframes
        kfs[0].header.stamp
        kf_stamps = [kf.header.stamp.to_sec() for kf in kfs]

        gt_stamps_in_range = [
            (t_w, t_o)
            for t_w, t_o in zip(self.gt_stamps_wall, self.gt_stamps_oxf)
            if min(kf_stamps) <= t_w <= max(kf_stamps)
        ]

        # if len(gt_stamps_in_range) < len(self.gt_stamps_wall):
        if len(gt_stamps_in_range) < 1:
            rospy.logwarn(
                f"Only {len(gt_stamps_in_range)/len(self.gt_stamps_wall)*100:.0f}% of gt images in range of keyframes"
            )
            return
        else:
            rospy.loginfo(
                f"{len(gt_stamps_in_range)/len(self.gt_stamps_wall)*100:.0f}% of gt images in range of keyframes"
            )

        if target_dir is not None:
            self.default_target = target_dir.data
        os.makedirs(self.default_target, exist_ok=True)
        self.mask_output = os.path.join(self.default_target, "semseg_preds")
        os.makedirs(self.mask_output, exist_ok=True)
        self.json_output = os.path.join(self.default_target, "semseg_preds.json")
        rospy.loginfo(
            f"Reprojecting landmarks for {len(gt_stamps_in_range)} gt images and storing to {self.default_target}"
        )

        # update landmarks
        self.landmarklayer.graph_changed_callback(None)  # type: ignore

        self.cam_poses = []
        self.cam_pose_labels = []
        self.annotations = []

        results = []
        idx = 1
        for stamp_wall, stamp_oxf in gt_stamps_in_range:
            if idx % 10 == 0:
                rospy.loginfo(f"Reprojecting {idx}/{len(gt_stamps_in_range)}")
            idx += 1

            # find index of closest keyframe to gt_stamp
            closest_kf_idx = min(
                range(len(kf_stamps)), key=lambda i: abs(kf_stamps[i] - stamp_wall)
            )

            # find global pose of closest keyframe
            closest_kf = kfs[closest_kf_idx]
            tf_closest_kf = ros_numpy.numpify(closest_kf.odom)

            # find relative pose of closest keyframe to gt_stamp
            try:
                t, r = self.tl.lookupTransformFull(
                    f"robotcar_{self.agent_no}/base_link",
                    closest_kf.header.stamp,
                    f"robotcar_{self.agent_no}/base_link",
                    rospy.Time.from_sec(stamp_wall),
                    f"robotcar_{self.agent_no}/odom",
                )
            except Exception as e:
                rospy.logwarn(f"Failed to lookup transform: {e}")
                continue
            tf_gtstamp2kfstamp = self.tl.fromTranslationRotation(t, r)

            tf_baselink2world = tf_closest_kf @ tf_gtstamp2kfstamp
            self.cam_poses.append(tf_baselink2world)
            self.cam_pose_labels.append(str(stamp_oxf) + ".png")

            position = tf_baselink2world[:3, 3]

            tf_world2baselink = np.linalg.inv(tf_baselink2world)
            tf_world2cam = self.tf_baselink2cam @ tf_world2baselink

            img = self.load_gt_img(stamp_oxf)
            vis_img, segments_bitmap, segments_info = self.reproject_landmarks(
                img, tf_world2cam, position
            )

            self.annotations.append(
                {
                    "segments_info": segments_info,
                    "file_name": str(stamp_oxf) + ".png",
                    "image_id": str(stamp_oxf),
                }
            )

            self.write_results(self.annotations, vis_img, segments_bitmap, stamp_oxf)

        rospy.loginfo(f"Reprojection done, wrote output for {len(self.annotations)} images")

        self.publish_poses()

    def publish_poses(self):
        pose_array = PoseArray()
        pose_array.header.frame_id = "world"
        pose_array.header.stamp = rospy.Time.now()
        pose_array.poses = [ros_numpy.msgify(Pose, p) for p in self.cam_poses]
        self.debug_pose_pub.publish(pose_array)

        ma = MarkerArray()
        for i, p in enumerate(self.cam_poses):
            m = Marker()
            m.header.frame_id = "world"
            m.ns = "gt_frame_pose_labels"
            m.id = i
            m.type = m.TEXT_VIEW_FACING
            m.scale.z = 1.0
            m.pose.position.x = p[0, 3]
            m.pose.position.y = p[1, 3]
            m.pose.position.z = p[2, 3] + 1.0
            m.color.r = 1.0
            m.color.g = 1.0
            m.color.b = 1.0
            m.color.a = 1.0
            m.text = f"{self.cam_pose_labels[i]}"
            ma.markers.append(m)

        self.debug_marker_pub.publish(ma)

    def write_results(self, annotations, vis_img, segments_bitmap, stamp):
        with open(self.json_output, "w") as f:
            json.dump({"annotations": annotations}, f)

        cv2.imwrite(f"{self.mask_output}/{stamp}.vis.png", vis_img)

        cv2.imwrite(f"{self.mask_output}/{stamp}.png", segments_bitmap)

    def masks_to_coco_fmt(self, masks: List[NDArray]) -> NDArray:
        # convert masks to coco format (ids<255 in red channel)
        assert len(masks) < 255, "Too many masks"
        segmentation_bitmap = np.zeros(
            (masks[0].shape[0], masks[0].shape[1], 3), dtype=np.uint8
        )

        for i, mask in enumerate(masks):
            # write id in red channel
            segmentation_bitmap[:,:,2][mask > 0] = i + 1

        assert (
            len(masks) == 0 or segmentation_bitmap.max() > 0
        ), "No masks in segmentation bitmap"

        return segmentation_bitmap

    def load_gt_img(self, stamp):
        img_path = self.gt_imgs + f"/{stamp}.png"
        img = cv2.imread(img_path)
        return img

    def reproject_landmarks(
        self, img, tf_world2cam: NDArray, position: NDArray
    ) -> Tuple[NDArray, NDArray, List[Dict]]:

        with self.landmarklayer.lock:
            landmarks = self.landmarklayer.get_nodes()

            close_landmarks: List[LandMarkNode] = []
            for lm in landmarks:
                lm_pos = lm.global_position
                dist = np.linalg.norm(position - lm_pos)
                if dist < 70.0:
                    close_landmarks.append(lm)

            if len(close_landmarks) == 0:
                segmentation_bitmap = np.zeros(
                    (img.shape[0], img.shape[1], 3), dtype=np.uint8
                )
                return img, segmentation_bitmap, []

            overlay = np.zeros(img.shape, dtype=np.uint8)

            masks = []
            segments_info = []
            id = 1

            for lm in close_landmarks:
                # reproject
                assert lm.global_position is not None and lm.global_cloud is not None

                lm_pos = np.reshape(lm.global_position, (-1, 1))
                # transform to camera frame
                lm_pos = np.dot(tf_world2cam, np.vstack([lm_pos, [1]]))[:3]
                lm_cloud = np.dot(
                    tf_world2cam,
                    np.vstack([lm.global_cloud.T, np.ones(lm.global_cloud.shape[0])]),
                ).T[:, :3]

                center_uv, _, _ = self.model.project(lm_pos.reshape(-1, 3).T, img.shape)
                pts_uv, depths, _ = self.model.project(lm_cloud.T, img.shape)
                if center_uv.shape[1] == 0 or pts_uv.shape[1] == 0:
                    continue

                mask = draw_mask(img, pts_uv)

                # occlusion checking moved to post processing
                # occl_ratio, occl_img = self.check_occlusion(mask, tf_world2cam, depths)

                segment_info = {
                    "id": id,
                    # "occlusion_ratio": occl_ratio,
                    "area": int(mask.sum()),
                    "depth": np.mean(depths),
                    "iscrowd": 0,
                    "n_pts": len(lm.global_cloud),
                    "obs_kfs": [obs.kf_id for obs in lm.observations],
                    "category_id": self.category2id[lm.class_name],
                    "category_name": lm.class_name,
                    "certainty": lm.avg_certainty(),
                    "n_obs": len(lm.observations),
                }
                masks.append(mask)

                # draw observation
                # if occl_ratio > 0.5:
                #     color = [0.5, 0.5, 0.5]
                # else:
                #     color = lm.rgba[:3]
                color = lm.rgba[:3]
                overlay = overlay + draw_landmark(
                    img, center_uv, pts_uv, depths, color, occl_ratio=None
                )
                # overlay = cv2.addWeighted(overlay, 1.0, occl_img, 0.5, 0.0)

                segments_info.append(segment_info)
                id += 1

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            vis_img = cv2.addWeighted(img_gray, 0.5, overlay, 1.0, 0)

            if len(masks) > 0:
                masks_coco = self.masks_to_coco_fmt(masks)
            else:
                masks_coco = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint)

            return vis_img, masks_coco, segments_info

    def check_occlusion(self, mask, tf_world2cam, lm_depths) -> Tuple[float, NDArray]:
        assert self.global_map is not None
        # translate into homogenous coordinates, then transform to camera frame
        # xyzw.shape = (4,N)
        map_pts_xyzw = np.vstack(
            (self.global_map.T, np.ones((self.global_map.shape[0])))
        )
        map_pts_cam_frame = (tf_world2cam @ map_pts_xyzw).T[:, :3]
        close_pts = np.linalg.norm(map_pts_cam_frame, axis=1) < 100.0
        map_pts_cam_frame = map_pts_cam_frame[close_pts]
        # project into image
        map_pts_uv, map_depths, _ = self.model.project(map_pts_cam_frame.T, mask.shape)
        # map_pts_uv.shape = (N,2)
        map_pts_uv = map_pts_uv.astype(int).T

        ### draw depth img for debugging
        # depth_img = np.zeros(mask.shape, dtype=np.uint8)
        # maxdepth = map_depths.max()
        # mindepth = map_depths.min()
        # for uv, depth in zip(map_pts_uv, map_depths):
        #     if depth < 4.0:
        #         continue
        #     u, v = uv
        #     h, w = self.model.get_projected_size(self.global_map_resolution, self.global_map_resolution, depth)
        #     # draw rectagle around point
        #     h2 = int(h//2)
        #     w2 = int(w//2)
        #     cv2.rectangle(depth_img, (u-w2, v-h2), (u+w2, v+h2), (1-(depth/maxdepth))*125+130, -1)

        # cv2.addWeighted(mask*255, 1.0, depth_img, 0.5, 0, depth_img)
        # cv2.imwrite(self.default_target + '/depth_img.png', depth_img)

        depth_selector = np.logical_and(
            map_depths < lm_depths.min() - 0.3, map_depths > 4.0
        )
        map_depths_infront = map_depths[depth_selector]
        map_pts_uv_infront = map_pts_uv[depth_selector]

        mask_pts_infront = np.zeros(mask.shape, dtype=np.uint8)
        maxdepth = map_depths.max()
        for uv, depth in zip(map_pts_uv_infront, map_depths_infront):
            if depth < 4.0:
                continue
            u, v = uv
            h, w = self.model.get_projected_size(
                self.global_map_resolution, self.global_map_resolution, depth
            )
            # draw rectagle around point
            h2 = int(h // 4)
            w2 = int(w // 4)
            cv2.rectangle(mask_pts_infront, (u - w2, v - h2), (u + w2, v + h2), 1, -1)

        occluded = mask * mask_pts_infront
        occlusion_ratio = occluded.sum() / mask.sum()

        ### render occlusion, mask infront and mask as seperate channels in rgb
        occlusion_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        occlusion_img[:, :, 0] = occluded * 255
        # occlusion_img[:,:,1] = mask_pts_infront * 255
        # occlusion_img[:,:,2] = occluded * 255

        return occlusion_ratio, occlusion_img
