import os
from typing import List, Tuple
from numpy.typing import NDArray
import rospy
from tokenize_anything import model_registry
from open_graph.amg_class import MyAutomaticMaskGenerator
import os
import cv2
from pathlib import Path
from tqdm import tqdm
from ram.models import tag2text
from groundingdino.util.inference import Model as GroundingDinoModel


class OpenGraphSemSeg:
    def __init__(self):
        weights_dir: str = rospy.get_param("~weights_folder")  # type: ignore
        concept_weights: str = rospy.get_param("~concept_weights")  # type: ignore
        use_tag2text: bool = rospy.get_param("~use_tag2text")  # type: ignore

        rospy.loginfo(f"Loading models from {weights_dir}")
        rospy.loginfo(f"Using tag2text" if use_tag2text else "Not using tag2text")

        tag2text_path = os.path.join(weights_dir, "tag2text_swin_14m.pth")
        gd_config_path = os.path.join(weights_dir, "GroundingDINO_SwinT_OGC_config.py")
        gd_weights_path = os.path.join(weights_dir, "groundingdino_swint_ogc.pth")
        tap_path = os.path.join(weights_dir, "tap_vit_l_v1_1.pkl")
        tap_concept_weights = os.path.join(weights_dir, concept_weights)
        paths = [gd_config_path, gd_weights_path, tap_path, tap_concept_weights]

        if use_tag2text:
            paths += [tag2text_path]

        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File {path} not found")

        delete_tag_index = list(range(3012, 3429))
        # load model
        if use_tag2text:
            self.tagging_model = (
                tag2text(
                    pretrained=tag2text_path,
                    image_size=384,
                    vit="swin_b",
                    delete_tag_index=delete_tag_index,
                )
                .eval()
                .to("cuda")
            )
        else:
            self.tagging_model = None

        self.grounding_dino_model = GroundingDinoModel(
            model_config_path=gd_config_path, model_checkpoint_path=gd_weights_path, device="cuda"
        )

        # TAP
        self.tap_model = model_registry["tap_vit_l"](checkpoint=tap_path)
        self.tap_model.concept_projector.reset_weights(tap_concept_weights)
        self.tap_model.text_decoder.reset_cache(max_batch_size=1000)

        self.mask_generator = MyAutomaticMaskGenerator(
            tagging_model=self.tagging_model,
            grounding_dino_model=self.grounding_dino_model,
            tap_model=self.tap_model,
        )

    def process_batch(self, images: List[NDArray]) -> List[Tuple]:
        results = []
        for img in images:
            res = self.mask_generator.generate(img, save_vis=False)
            results.append(res)
        return results
