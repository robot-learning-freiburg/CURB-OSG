import open3d as o3d
from curb_projection.msg._StaticObjectObsArray import StaticObjectObsArray
from geometry_msgs.msg import Point
import ros_numpy
import rospy
import numpy as np
from torch import Tensor
from typing import List, Tuple, Union
from numpy.typing import NDArray
from std_msgs.msg import Header
from tf import ExtrapolationException, LookupException, TransformListener  # type: ignore
from .utils import BatchSample, ColorGen, ModelsStore
from curb_projection.msg import StaticObjectObs
import sensor_msgs.point_cloud2 as pcl2


class SemanticKeyframe:
    def __init__(
        self,
        sample: BatchSample,
        models: ModelsStore,
        tl: TransformListener,
    ):
        self.tl = tl
        self.kf = sample.kf
        self.all_points = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(self.kf.cloud)

        self.models = models

        # params
        self.global_debug_map: bool = rospy.get_param("~global_debug_map", False)  # type: ignore
        self.apply_outlier_filtering: bool = rospy.get_param("~outlier_filtering", False)  # type: ignore
        self.apply_range_filter: bool = rospy.get_param("~range_filtering", False)  # type: ignore
        self.min_obj_score: float = rospy.get_param("~min_obj_score", 0.0)  # type: ignore
        rospy.loginfo_once(f"outlier_filtering: {self.apply_outlier_filtering}, range_filtering: {self.apply_range_filter}, min_obj_score: {self.min_obj_score}")

        self.target_classes: List[str] = rospy.get_param("~target_classes").split(",")  # type: ignore

        self.found_objects: List[StaticObjectObs] = []

        # copy some metadata
        self.stamp = sample.kf.cloud.header.stamp
        self.frame_id = sample.kf.cloud.header.frame_id
        self.id = sample.kf.id
        self.agent_no = sample.kf.agent_no

        # storage for global debug map
        self.all_clouds = []
        self.all_classes = []

        # project all cams to the keyframe points
        assert sample.is_processed()
        for cam_idx in range(sample.n_cams):
            img = sample.imgs[cam_idx]
            img_header = sample.img_headers[cam_idx]

            points, uv_map = self.run_projection(img.shape, img_header)

            masks, mask_classes, mask_scores = sample.results[cam_idx]

            obj_clouds, obj_classes, obj_scores = self.find_masked_points(
                points, uv_map, masks, mask_classes, mask_scores
            )

            obj_clouds, obj_classes, obj_scores = self.filter_objects(
                obj_clouds, obj_classes, obj_scores
            )

            self.add_found_objects(obj_clouds, obj_classes, obj_scores)

            # if self.id in range(1, 20) or self.id in range(1000001, 1000020):
            #     self.reproject_points(
            #         cam_idx, sample.get_img(cam_idx), sample.img_headers[cam_idx]
            #     )
        
        # rospy.logdebug(f"Semantic Mapping found {len(self.found_objects)} objects in keyframe {self.id}")

    def find_masked_points(
        self,
        points: NDArray,
        uv_map: NDArray,
        masks: List[NDArray],
        classes: List[str],
        scores: List[float],
    ) -> Tuple[List[NDArray], List[str], List[float]]:
        """Extract points from the point cloud that are within each mask if the class is in the set of target classes."""
        clouds = []
        obj_classes = []
        obj_scores = []
        for i in range(len(masks)):
            if not self.global_debug_map and not classes[i] in self.target_classes:
                continue

            mask = masks[i]
            points_mask = mask[uv_map[1], uv_map[0]]
            obj_cloud = points[points_mask == 1]

            if self.global_debug_map:
                self.all_clouds.append(obj_cloud)
                self.all_classes.append(classes[i])

            if not classes[i] in self.target_classes:
                continue

            if len(obj_cloud) < 2:
                continue

            clouds.append(obj_cloud)
            obj_classes.append(classes[i])
            obj_scores.append(scores[i])
        return clouds, obj_classes, obj_scores

    def filter_objects(
        self,
        obj_clouds: List[NDArray],
        obj_classes: List[str],
        obj_scores: List[float],
    ) -> Tuple[List[NDArray], List[str], List[float]]:

        out_clouds = []
        out_classes = []
        out_scores = []

        for in_cloud, in_class, in_score in zip(obj_clouds, obj_classes, obj_scores):

            if in_score < self.min_obj_score:
                rospy.logdebug(
                    f"score below threshold ({in_score}/{self.min_obj_score}) of instance of class {in_class}"
                )
                continue

            # apply open3d outlier removal
            if self.apply_outlier_filtering:
                out_cloud = self.outlier_filter(in_cloud, 6, 0.6)

                if out_cloud.shape[0] < 2:
                    rospy.logdebug(
                        f"less than two points left after outlier removal of instance of class {in_class}"
                    )
                    continue

            if self.apply_range_filter:
                out_cloud = self.range_filter(in_cloud, 0.6)
                if out_cloud.shape[0] < 2:
                    rospy.logdebug(
                        f"less than two points left after range filtering of instance of class {in_class}"
                    )
                    continue
            
            out_clouds.append(out_cloud)
            out_classes.append(in_class)
            out_scores.append(in_score)
        
        return out_clouds, out_classes, out_scores

    def outlier_filter(
        self,
        cloud: NDArray,
        nb_points: int,
        radius: float
    ) -> NDArray:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=3, std_ratio=0.5)
        cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
        out_cloud = np.asarray(cl.points)
        return out_cloud
    
    def range_filter(
        self,
        cloud: NDArray,
        min_range: float
    ) -> NDArray:
        dists = np.linalg.norm(cloud, axis=1)
        min_dist = np.min(dists)
        offsets = np.abs(dists - min_dist)
        mask = offsets <= min_range
        out_cloud = cloud[mask]
        return out_cloud

    def add_found_objects(
        self,
        obj_clouds: List[NDArray],
        obj_classes: List[str],
        obj_scores: List[float]
    ):
        for i in range(len(obj_clouds)):
            # now we can create the object
            obj = StaticObjectObs()
            obj.header = Header()
            obj.header.stamp = self.stamp
            obj.header.frame_id = f"keyframe/{self.id}"
            obj.centroid = ros_numpy.msgify(Point, np.mean(obj_clouds[i], axis=0))
            obj.certainty = obj_scores[i]
            # clip feature disabled
            # obj.clip_feature = point_features[instance_mask].mean(axis=0)
            obj.cloud = pcl2.create_cloud_xyz32(self.kf.header, obj_clouds[i])
            obj.keyframe_id = self.id
            obj.observing_agent_id = self.agent_no
            obj.instance_id = len(self.found_objects) + 1
            obj.class_name.data = obj_classes[i]

            self.found_objects.append(obj)

    def run_projection(
        self,
        img_shape: Tuple,
        img_header: Header,
    ) -> Tuple[NDArray, NDArray]:
        # transformation into camera frame
        translation, rotation = self.tl.lookupTransform(
            img_header.frame_id, self.kf.cloud.header.frame_id, rospy.Time(0)
        )
        lidar_to_cam = self.tl.fromTranslationRotation(translation, rotation)

        # transform time difference
        base_link_frame = f"robotcar_{self.kf.agent_no}/base_link"
        try:
            t, r = self.tl.lookupTransformFull(
                base_link_frame,
                img_header.stamp,
                base_link_frame,
                self.kf.header.stamp,
                f"/robotcar_{self.kf.agent_no}/odom",
            )
            tf_kf2imagetime = self.tl.fromTranslationRotation(t, r)
        except ExtrapolationException as e:
            rospy.logwarn("maskclip projection node: %s", repr(e))
            tf_kf2imagetime = np.eye(4)

        # translate into homogenous coordinates, then transform to camera frame and image time
        xyz = self.all_points
        # xyzw.shape = (4,N)
        xyzw = np.vstack((xyz.T, np.ones((self.all_points.shape[0]))))
        xyzw_in_cam_frame = lidar_to_cam @ tf_kf2imagetime @ xyzw

        model = self.models.get_model(img_header.frame_id)

        # uv: (N,2) array of integer pixel coordinates in the original image
        # correspondences: maps uv indices back to indices in the point cloud
        uv, _, correspondences = model.project(xyzw_in_cam_frame, img_shape[:2])
        uv = uv.astype(np.int32)

        pts = xyz[correspondences]

        return pts, uv

    # def reproject_points(self, img, img_header: Header, uv: NDArray, pts: NDArray):
    #     uv = self.uvs[cam_idx].astype(int)
    #     classes = self.classes[cam_idx]
    #     colors = [COLORS[c] for c in classes]
    #     pts = self.proj_points[cam_idx]

    #     overlay = np.zeros(img.shape, dtype=np.uint8)
    #     overlay[:, :, :] = img[:, :, :]

    #     for i, rgb in enumerate(colors):
    #         rad = int(15 / np.sqrt(np.linalg.norm(pts[i])))
    #         rad = max(rad, 1)
    #         rad = min(rad, 10)

    #         cv2.circle(overlay, uv[:, i], rad, colors[i] * 255, -1)  # type: ignore

    #     class_names = self.prompt_str.split(",")

    #     for i, c in enumerate(class_names + [img_header.frame_id]):
    #         cv2.putText(  # type: ignore
    #             overlay,
    #             c,
    #             (10, 20 * (i + 1)),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             1,
    #             COLORS[i] * 255.0,  # type: ignore
    #             3,
    #             2,
    #         )

    #     cv2.imwrite(f"imgdebug/{self.agent_no}_{self.id}_{cam_idx}.png", overlay)

    def get_found_objects(self) -> StaticObjectObsArray:
        msg = StaticObjectObsArray()
        msg.observations = self.found_objects
        return msg

    def get_xyzrgb(self, color_gen: ColorGen, in_world=True) -> Union[None, NDArray]:
        all_points = np.ndarray((0, 3))
        rgb = np.ndarray((0,3))
        for cloud, class_ in zip(self.all_clouds, self.all_classes):
            all_points = np.vstack([all_points, cloud])
            color = color_gen.get_color(class_)
            rgb = np.vstack([rgb, np.tile(color, (len(cloud), 1))])
        
        if not in_world:
            return np.c_[all_points, rgb]
        
        try:
            translation, rotation = self.tl.lookupTransform(
                "world", f"keyframe/{self.id}", rospy.Time(0)
            )
        except LookupException as e:
            rospy.logdebug(repr(e))
            return None
        
        keyframe_to_world = self.tl.fromTranslationRotation(translation, rotation)
        # perform the transformation in homogenous coordinates
        xyzw = np.c_[all_points, np.ones((all_points.shape[0],))]
        xyz = (keyframe_to_world @ xyzw.T).T[:, :3]
        return np.c_[xyz, rgb]