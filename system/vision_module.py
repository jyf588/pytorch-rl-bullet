import cv2
import imageio
import numpy as np
from typing import *
from argparse import Namespace

import torch
import torchvision.transforms as transforms

import ns_vqa_dart.bullet.seg
from ns_vqa_dart.bullet import dash_object, gen_dataset
from ns_vqa_dart.scene_parse.attr_net.model import get_model
from ns_vqa_dart.scene_parse.attr_net.options import BaseOptions


class VisionModule:
    def __init__(self, load_checkpoint_path: str):
        options = self.get_options(load_checkpoint_path=load_checkpoint_path)
        self.model = get_model(options)
        self.model.eval_mode()

    def get_options(self, load_checkpoint_path: str):
        """Creates the options namespace to define the vision model."""
        options = Namespace(
            inference_only=True,
            load_checkpoint_path=load_checkpoint_path,
            gpu_ids="0",
            concat_img=True,
            with_depth=False,
            fp16=False,
        )
        options = BaseOptions().parse(opt=options, save_options=False)
        return options

    def predict(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Runs inference on an image to get vision predictions.

        Args:
            rgb: The RGB image of the scene.
            mask: The segmentation mask of the object to predict.
        
        Returns:
            pred: The model prediction for the object.
        """
        # Create the input X data to the model. We create the data tensor even
        # if the mask is empty (i.e., object is completely occluded)
        data = dash_object.compute_X(img=rgb, mask=mask, keep_occluded=True)

        # Debugging
        # debug_seg = data[:, :, :3]
        # debug_rgb = data[:, :, 3:6]
        # input_debug = np.hstack([debug_seg, debug_rgb])
        # path = f"/home/michelle/tmp/vision_input_{oid}.png"
        # imageio.imwrite(path, input_debug)
        # print(f"Wrote debug image to: {path}")

        # Predict.
        pred = self.predict_from_data(data=data)[0]
        return pred

    def predict_from_data(self, data: np.ndarray):
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
