import os
import imageio
import numpy as np
from PIL import Image


for i in range(340, 414):
    pathA = f"/home/mguo/data/dash/seg_tiny/stack/rgb/0000/{i:06}.png"
    pathB = f"/home/mguo/data/dash/system_seg_tiny/stack/rgb/0000/{i:06}.png"

    if not os.path.exists(pathB):
        continue

    imgA = Image.open(pathA)
    imgB = Image.open(pathB)

    imgA_rgba = imgA.convert("RGBA")
    imgB_rgba = imgB.convert("RGBA")

    blend = Image.blend(imgA_rgba, imgB_rgba, 0.5)
    blend.save(f"/home/mguo/blend/{i:06}.png", "PNG")

    imgA = imageio.imread(pathA)
    imgB = imageio.imread(pathB)

    mask = imgA != imgB

    mask_2d = mask.sum(axis=2)
    num_diff_pixels = (mask_2d > 0).sum()

    # print(mask.dtype)
    # diff = mask.sum()
    # print(mask.shape)
    print(num_diff_pixels)

# diff = (imgA - imgB).sum()
# print(diff)
