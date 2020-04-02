from argparse import Namespace
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import *

from ns_vqa_dart.bullet import dash_object, gen_dataset
from ns_vqa_dart.scene_parse.attr_net.model import get_model
from ns_vqa_dart.scene_parse.attr_net.options import BaseOptions


class VisionModule:
    def __init__(self):
        options = self.get_options()
        self.model = get_model(options)
        self.model.eval_mode()

    def get_options(self):
        """Creates the options namespace to define the vision model."""
        options = Namespace(
            inference_only=True,
            load_checkpoint_path="/home/michelle/mguo/outputs/dash_v002_5K/checkpoint_best.pt",
            gpu_ids="0",
            concat_img=True,
            with_depth=False,
            fp16=False,
        )
        options = BaseOptions().parse(opt=options, save_options=False)
        return options

    def predict(
        self, oid: int, rgb: np.ndarray, seg_img: np.ndarray
    ) -> np.ndarray:
        """Runs inference on an image to get vision predictions.

        Args:
            rgb: The RGB image of the scene.
            seg: The segmentation of the scene.
        
        Returns:
            pred: The model predictions.
        """
        seg = gen_dataset.seg_img_to_map(seg_img)
        data = dash_object.compute_X(
            oid=oid, img=rgb, seg=seg, keep_occluded=False
        )
        data_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 6, std=[0.225] * 6),
            ]
        )
        X = torch.zeros(size=(1, 6, 480, 480))
        X[0] = data_transforms(data)
        self.model.set_input(X)
        self.model.forward()
        pred = self.model.get_pred()
        return pred
