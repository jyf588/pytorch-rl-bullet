import os
import cv2
import imageio
import numpy as np
from typing import *
from argparse import Namespace

import torch
import torchvision.transforms as transforms

import ns_vqa_dart.bullet.seg
from ns_vqa_dart.bullet import dash_object
from ns_vqa_dart.scene_parse.attr_net.model import get_model
from ns_vqa_dart.scene_parse.attr_net.options import BaseOptions


class VisionModule:
    def __init__(self, load_checkpoint_path: str, debug_dir: str):
        self.debug_dir = debug_dir
        options = self.get_options(load_checkpoint_path=load_checkpoint_path)
        self.model = get_model(options)
        self.model.eval_mode()

        self.data_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 6, std=[0.225] * 6),
            ]
        )

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

    def predict(self, rgb: np.ndarray, masks: np.ndarray, debug_id: int) -> np.ndarray:
        """Runs inference on an image to get vision predictions.

        Args:
            rgb: The RGB image of the scene.
            masks: The segmentation masks of objects to predict with shape 
                (N, H, W)
        
        Returns:
            pred: The model prediction for the object.
        """
        data = []
        _, W, _ = rgb.shape

        # Construct the input data for each mask.
        for mask in masks:
            # Create the input X data to the model. We create the data tensor even
            # if the mask is empty (i.e., object is completely occluded)
            X = dash_object.compute_X(img=rgb, mask=mask, keep_occluded=True)
            data.append(X)

        # Writing images for debugging.
        if self.debug_dir is not None:
            self.gen_debug_images(data=data, debug_id=debug_id)

        # Predict.
        data = np.array(data)
        assert data.shape == (len(masks), W, W, 6)  # (N, W, W, 6)
        pred = self.predict_from_data(data=data)
        return pred

    def predict_from_data(self, data: np.ndarray):
        """
        Args:
            data: A numpy array with shape (N, H, W, 6).
        """
        N, _, W, C = data.shape
        X = torch.zeros(size=(N, C, W, W))

        # Normalize and convert to pytorch tensor. We perform this per example
        # because that's what the data transform expects.
        for i in range(N):
            X[i] = self.data_transforms(data[i])

        # Run the model.
        self.model.set_input(X)
        self.model.forward()
        pred = self.model.get_pred()
        return pred

    def gen_debug_images(self, data: List[np.ndarray], debug_id: int):
        """
        Args:
            data: A list of data tensors, one for each object.
        """
        rows = []
        for object_data in data:
            rows.append(np.hstack([object_data[:, :, :3], object_data[:, :, 3:6]]))
        image = np.vstack(rows)
        path = os.path.join(self.debug_dir, f"{debug_id:04}", "vision_input.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.imwrite(path, image)
        print(f"Wrote debug image to: {path}")
