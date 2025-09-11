import pprint
from time import time, perf_counter
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import rospy
import torch
import cv2
from torch import Tensor
import torch.nn.functional as F
from featup.featurizers.maskclip.clip import tokenize
from featup.util import pca, remove_axes, unnorm
from numpy.typing import ArrayLike, NDArray
from pytorch_lightning import seed_everything
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from scipy.ndimage import zoom
from scipy.special import softmax

pp = pprint.PrettyPrinter(indent=4)

seed_everything(0)


class MaskCLIPSemSeg:

    def __init__(
        self,
    ) -> None:

        if not torch.cuda.is_available():
            rospy.logerr("Cuda device not found")
            exit(1)

        self.device = "cuda"
        self.model = (
            torch.hub.load("mhamilton723/FeatUp", "maskclip", use_norm=False)
            .eval()
            .to(self.device)
        )

        self.patch_size = 16
        self.feature_size = 512
        self.patch_h = 20
        self.patch_w = 26

        self.transform_rect = Compose(
            [
                ToTensor(),  # HxWxC with [0, 255] -> CxHxW [0.0, 1.0]
                Resize(
                    (self.patch_size * self.patch_h, self.patch_size * self.patch_w),
                    antialias=False,
                ),  # Default is bilinear
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # square transform for the 1024x1024 mono images
        self.transform_square = Compose(
            [
                ToTensor(),  # HxWxC with [0, 255] -> CxHxW [0.0, 1.0]
                Resize(
                    (self.patch_size * self.patch_h, self.patch_size * self.patch_h),
                    antialias=False,
                ),  # Default is bilinear
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.fit_pca = None

        rospy.loginfo(f"Model loaded: MaskClip")

        self.prompt_str: str = rospy.get_param("~maskclip_prompt")  # type: ignore
        self.class_names = self.prompt_str.split(",")
        self.clip_prompt = torch.vstack(
            [
                torch.tensor(self.clip_encode(c, normalize=True))
                for c in self.class_names
            ]
        ).to(self.device)

    def process_batch(
        self, images: List[NDArray], upsample: bool = False
    ) -> List[Tuple]:
        imgsize = images[0].shape
        if imgsize == (1024, 1024, 3):
            transform = self.transform_square
        elif imgsize == (960, 1280, 3):
            transform = self.transform_rect
        else:
            rospy.logerr(f"Unknown image size encountered: {imgsize}")
            exit(1)

        batch_tens = torch.stack([transform(img) for img in images]).to(self.device)
        with torch.no_grad():
            features = self.model(batch_tens)
            features = features.cpu()

        if upsample:
            # upsampling takes longer than running the model!
            features = F.interpolate(
                features, imgsize[:2], mode="bilinear", align_corners=False
            )
        # features = [f.numpy() for f in torch.unbind(features.permute(0, 2, 3, 1))]
        features = torch.unbind(features.permute(0, 2, 3, 1))

        results = []
        for f in features:
            classes, certainties = self.classify_img_pixels(imgsize, f)
            result = self.component_segmentation(classes, certainties)
            results.append(result)

        return results

    def get_image_features(
        self, image: ArrayLike, upsample: bool = False, return_pca: bool = False
    ) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]:
        # Pre-process image and add batch dimension
        torch_image = self.transform(image).unsqueeze(0).to(self.device)

        # a = time()
        with torch.no_grad():
            if self.use_featup:
                features = self.model(torch_image)
            else:
                features = self.model.model(torch_image)
        # print(time() - a, torch_image.shape)
        # Feature size: BxCxHxW

        if return_pca:
            [pca_features], fit_pca = pca([features], dim=3, fit_pca=self.fit_pca)
            if self.fit_pca is None:  # Only fit PCA once
                self.fit_pca = fit_pca
        else:
            pca_features = None

        if upsample:
            features = F.interpolate(
                features, image.shape[:2], mode="bilinear", align_corners=False
            )
            if pca_features is not None:
                pca_features = F.interpolate(
                    pca_features, image.shape[:2], mode="bilinear", align_corners=False
                )

        features = features.permute(0, 2, 3, 1).squeeze().cpu().numpy()

        if pca_features is not None:
            pca_features = pca_features.permute(0, 2, 3, 1).squeeze().cpu().numpy()
            pca_features = (pca_features * 255.0).astype(np.uint8)  # Convert to RGB

        return (features, pca_features) if return_pca else features

    def clip_encode(self, prompt: str, normalize=False) -> torch.Tensor:
        with torch.no_grad():
            tokens = tokenize(prompt).to(self.device)
            embedding = self.model.model.model.encode_text(tokens)
            embedding_np = embedding.detach().float()
        if normalize:
            embedding_np = embedding_np / torch.linalg.norm(embedding_np)
        return embedding_np.squeeze()

    def classify_img_pixels(
        self, target_shape: Tuple, img_features: Tensor
    ) -> Tuple[NDArray, NDArray]:
        M_classes = len(self.clip_prompt)
        assert M_classes > 0

        # divide each feature by its norm for cosine similarity
        img_features = img_features.cuda()
        img_features = img_features / torch.norm(img_features, dim=2, keepdim=True)

        feat_h, feat_w, _ = img_features.shape
        scores_per_class = (
            torch.einsum("ijk,lk->ijl", img_features, self.clip_prompt).cpu().numpy()
        )

        # extract classes
        classes = np.argmax(scores_per_class, axis=2)

        # extract a kind of 'certainty' for each prediction as the difference of
        # the predicted classes score to the mean score for each pixel.
        # class_scores = np.max(scores_per_class, axis=2)
        # mean_scores = np.mean(scores_per_class, axis=2)
        # certainty = class_scores - mean_scores
        # # normalize between (0,1)
        # certainty /= certainty.max()
        smax = softmax(scores_per_class, axis=2)
        certainty = np.max(smax, axis=2)
        # discrete upsampling with nearest-neighbor
        img_h, img_w, _ = target_shape
        img_classes = zoom(classes, (img_h / feat_h, img_w / feat_w), order=0)
        certainty = zoom(certainty, (img_h / feat_h, img_w / feat_w), order=0)

        return img_classes, certainty

    def component_segmentation(
        self, pixel_classes: NDArray, pixel_scores: NDArray
    ) -> Tuple[List, List, List]:
        masks = []
        classes = []
        certainties = []
        for class_id, class_name in enumerate(self.class_names):
            # masks.append(self.watershed_segment(img, img_classes, obj_class))
            fg = (pixel_classes == class_id).astype(np.uint8)
            _, inst_ids = cv2.connectedComponents(fg)
            for inst_id in range(1, inst_ids.max() + 1):
                mask = (inst_ids == inst_id).astype(np.uint8)
                masks.append(mask)
                classes.append(class_name)
                certainty = np.mean(pixel_scores[mask == 1])
                certainties.append(certainty)

        return (masks, classes, certainties)

    # Alternative segmentation method: unused
    #
    # def watershed_segmentation(
    #     self, img: NDArray, img_classes: NDArray, class_id: int
    # ) -> NDArray:

    #     fg = (img_classes == class_id).astype(np.uint8)

    #     # use distance transform + threshold to seperate touching object instances
    #     dist_transformed = cv2.distanceTransform(fg, cv2.DIST_L2, cv2.DIST_MASK_3)
    #     halfdist = int(dist_transformed.max() * 0.5)
    #     _, sure_fg = cv2.threshold(dist_transformed, halfdist, 255, cv2.THRESH_BINARY)
    #     sure_fg = sure_fg.astype(np.uint8)

    #     # dilate fg to find certain background
    #     kernel = np.ones((halfdist, halfdist), np.uint8)
    #     sure_bg = cv2.dilate(fg, kernel, iterations=3)

    #     # find unknown regions (not background or object)
    #     unknown = cv2.subtract(sure_bg, sure_fg)

    #     _, inst_markers = cv2.connectedComponents(sure_fg.astype(np.uint8))
    #     # background is marked as 0, but we need to mark it as 1
    #     inst_markers += 1
    #     inst_markers[unknown == 1] = 0

    #     inst_markers = cv2.watershed(img, inst_markers)

    #     return inst_markers
