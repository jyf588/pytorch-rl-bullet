"""Defines a segmentation module for evaluation only."""
import os
import imageio
import numpy as np

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


class SegmentationModule:
    def __init__(self, load_checkpoint_path: str, debug_dir: str):
        self.debug_dir = debug_dir

        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.99
        cfg.MODEL.WEIGHTS = load_checkpoint_path
        self.predictor = DefaultPredictor(cfg)

    def predict(self, img: np.ndarray, debug_id: int) -> np.ndarray:
        """Predicts binary instance segmentations of objects vs. background for
        a given image.

        Args:
            img: The RGB image to predict segmentations on.
            
        Returns:
            masks: A numpy array of shape (N, H, W) of instance masks.
        """
        # Write the input image for debugging.
        if self.debug_dir is not None:
            path = os.path.join(
                self.debug_dir, f"{debug_id:04}", "seg_input.png"
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            imageio.imwrite(path, img)

        outputs = self.predictor(img)

        # This produces a numpy array of shape (N, H, W) containing binary
        # masks.
        masks = outputs["instances"].to("cpu")._fields["pred_masks"].numpy()
        # seg = np.full((320, 480), -1, dtype=np.uint8)
        # for idx, mask in enumerate(masks):
        #     seg[mask] = idx
        #     mask_img = mask.astype(np.uint8) * 255
        #     path = f"/home/michelle/tmp/seg_mask_{idx}.png"
        #     imageio.imwrite(path, mask_img)
        #     print(f"Wrote mask image to: {path}")
        return masks
