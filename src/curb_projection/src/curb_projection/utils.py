from typing import Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray

from hdl_graph_slam.msg._Keyframe_msg import Keyframe_msg
from std_msgs.msg import Header
from .camera_model import CameraModel
import torch


def visualize_segments(img: NDArray, masks, classes, scores, save_path):
    raise NotImplementedError

            # debug output
            # vis = self.visualize_img_classes(img_classes, certainty, img)
            # cv2.imwrite(
            #     f"/workspaces/collaborative-scene-graphs/imgdebug/a{self.agent_no}_kf{self.id:07}_{cam_idx}_certainty.png",
            #     vis,
            # )
    # def visualize_img_classes(
    #     self, img_classes: NDArray, certainty: NDArray, img: NDArray
    # ):
    #     class_colors = np.array([[COLORS[c] for c in row] for row in img_classes])
    #     img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     img_gray = np.expand_dims(img_gray, axis=2)
    #     # certainty is of same shape as img, use as alpha
    #     assert certainty.shape == img.shape[:2]
    #     alpha = np.expand_dims(certainty, axis=2)
    #     img_colored = alpha * 255 * class_colors + ((1 - alpha) * img_gray)
    #     img_colored = img_colored.astype(np.float32)

    #     for i, c in enumerate(self.all_class_names):
    #         cv2.putText(  # type: ignore
    #             img_colored,
    #             c,
    #             (10, 20 * (i + 1)),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             fontScale=1,
    #             color=COLORS[i] * 255.0,  # type: ignore
    #             thickness=2,
    #             lineType=2,
    #         )

    #     return cv2.cvtColor(img_colored, cv2.COLOR_RGB2BGR)


class ModelsStore:
    def __init__(self, models_dir):
        self.cameras = ["stereo/left", "mono_right", "mono_rear", "mono_left"]
        self.models = []

        for cam in self.cameras:
            model = CameraModel(models_dir, cam)
            self.models.append(model)

    def get_model(self, camera_frame: str) -> CameraModel:
        if "stereo" in camera_frame:
            return self.models[0]
        elif "mono_right" in camera_frame:
            return self.models[1]
        elif "mono_rear" in camera_frame:
            return self.models[2]
        elif "mono_left" in camera_frame:
            return self.models[3]
        else:
            raise KeyError


class BatchSample:
    def __init__(
        self, kf: Keyframe_msg, img_headers: List[Header], imgs: List[NDArray]
    ):
        assert len(imgs) == len(img_headers)
        self.n_cams = len(imgs)

        self.kf = kf
        self.imgs = imgs
        self.img_headers = img_headers

        # map img index to feature result
        self.results: Dict[int, Tuple] = dict()

    def get_img(self, idx: int):
        return self.imgs[idx]

    def is_processed(self) -> bool:
        return len(self.results) == len(self.imgs)

    def add_result(self, cam_idx: int, masks: List[NDArray], classes: List[str], scores: List[float]):
        self.results[cam_idx] = (masks, classes, scores)


class Batch:
    def __init__(self, batch_size=5, n_cams=4):
        self.samples: List[BatchSample] = []
        self.batch_size = batch_size
        self.n_cams = n_cams

    def is_full(self) -> bool:
        return len(self.samples) == self.batch_size

    def is_processed(self) -> bool:
        return self.is_full() and all(s.is_processed() for s in self.samples)

    def add_sample(self, sample: BatchSample):
        assert not self.is_full()
        self.samples.append(sample)

    def get_imgs(self, cam_idx: int) -> List[NDArray]:
        """Get a list of images in the batch, each image comes from the same cam
        and has the same resolution

        Args:
            idx (int): camera index from [0,3]

        Returns:
            List[NDArray]: list of images in the batch from this cam
        """
        assert self.is_full() and cam_idx < self.n_cams
        return [sample.get_img(cam_idx) for sample in self.samples]

    def store_results(self, cam_idx: int, results: List[Tuple]):
        assert cam_idx < self.n_cams and len(results) == self.batch_size
        for i in range(self.batch_size):
            masks, classes, scores = results[i]
            self.samples[i].add_result(cam_idx, masks, classes, scores)


class ColorGen:
    def __init__(self):
        self.cityscapes_colors = {
            # from https://raw.githubusercontent.com/mcordts/cityscapesScripts/refs/heads/master/cityscapesscripts/helpers/labels.py
            'unlabeled'            : np.array([  0,  0,  0]),
            'ego vehicle'          : np.array([  0,  0,  0]),
            'rectification border' : np.array([  0,  0,  0]),
            'out of roi'           : np.array([  0,  0,  0]),
            'static'               : np.array([  0,  0,  0]),
            'dynamic'              : np.array([111, 74,  0]),
            'ground'               : np.array([ 81,  0, 81]),
            'road'                 : np.array([128, 64,128]),
            'sidewalk'             : np.array([244, 35,232]),
            'parking'              : np.array([250,170,160]),
            'rail track'           : np.array([230,150,140]),
            'building'             : np.array([ 70, 70, 70]),
            'wall'                 : np.array([102,102,156]),
            'fence'                : np.array([190,153,153]),
            'guard rail'           : np.array([180,165,180]),
            'bridge'               : np.array([150,100,100]),
            'tunnel'               : np.array([150,120, 90]),
            'pole'                 : np.array([153,153,153]),
            'polegroup'            : np.array([153,153,153]),
            'traffic light'        : np.array([250,170, 30]),
            'traffic sign'         : np.array([220,220,  0]),
            'vegetation'           : np.array([107,142, 35]),
            'terrain'              : np.array([152,251,152]),
            'sky'                  : np.array([ 70,130,180]),
            'person'               : np.array([220, 20, 60]),
            'rider'                : np.array([255,  0,  0]),
            'car'                  : np.array([  0,  0,142]),
            'truck'                : np.array([  0,  0, 70]),
            'bus'                  : np.array([  0, 60,100]),
            'caravan'              : np.array([  0,  0, 90]),
            'trailer'              : np.array([  0,  0,110]),
            'train'                : np.array([  0, 80,100]),
            'motorcycle'           : np.array([  0,  0,230]),
            'bicycle'              : np.array([119, 11, 32]),
            'license plate'        : np.array([  0,  0,142]),
        }
        self.colors = [
            np.array([0.176, 0.961, 0.192]),  # green
            np.array([0.016, 0.686, 0.91]),  # light blue
            np.array([0.098, 0.216, 0.62]),  # dark blue
            np.array([0.929, 0.016, 0.016]),  # bright red
            np.array([1.0, 0.655, 0.0]),  # bright orange
            np.array([0.467, 0.09, 0.541]),  # purple
            np.array([0.91, 0.91, 0.91]),  # light grey
            np.array([0.61, 0.61, 0.61]),  # dark grey
            np.array([0.988, 0.553, 0.38]),  # yellowish orange
            np.array([0.231, 0.671, 0.028]),  # forest green
            np.array([0.651, 0.047, 0.89]),  # dark purple
            np.array([0.592, 0.412, 0.757]),  # light purple
            np.array([0.184, 0.310, 0.196]),  # dark green
            np.array([0.843, 0.804, 0.1]),  # yellow
            np.array([1.0, 0.412, 0.557]),  # magenta
            np.array([0.0, 0.749, 1.0]),  # sky blue
            np.array([0.851, 0.325, 0.098]),  # orange
            np.array([0.2, 0.627, 0.447]),  # sea green
            np.array([0.788, 0.294, 0.737]),  # violet
            np.array([0.957, 0.643, 0.376]),  # peach
        ]
        self.class2color = dict()
    
    def get_color(self, class_name: str):
        if class_name in self.cityscapes_colors.keys():
            return self.cityscapes_colors[class_name] / 255.0  # normalize to [0,1]

        if class_name not in self.class2color.keys():
            self.class2color[class_name] = self.colors[len(self.class2color) % len(self.colors)]
        return self.class2color[class_name]

CITYSCAPES_CLASS_NAMES = [
    "road",
    "sidewalk",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "on rails",
    "motorcycle",
    "bicycle",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic sign",
    "traffic light",
    "vegetation",
    "terrain",
    "sky"
]