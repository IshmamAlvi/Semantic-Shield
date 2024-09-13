import sys
sys.path.append( '../../' )
from clip_vit.config import CFG
import numpy as np
import albumentations as A
import torch
import pandas as pd
import random

def get_transforms_noise_bpp(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [   
                A.Resize(CFG.size, CFG.size, always_apply=True),
            ]
        )

def rnd1(x, decimals, out):
    return np.round_(x, decimals, out)


def floydDitherspeed(image, squeeze_num=8):
    channel, h, w = image.shape
    for y in range(h):
        for x in range(w):
            old = image[:, y, x]
            temp = np.empty_like(old).astype(np.float64)
            new = rnd1(old / 255.0 * (squeeze_num - 1), 0, temp) / (squeeze_num - 1) * 255
            error = old - new
            image[:, y, x] = new
            if x + 1 < w:
                image[:, y, x + 1] += error * 0.4375
            
            if (y + 1 < h) and (x + 1 < w):
                image[:, y + 1, x + 1] += error * 0.0625
            if y + 1 < h:
                image[:, y + 1, x] += error * 0.3125
            if (x - 1 >= 0) and (y + 1 < h):
                image[:, y + 1, x - 1] += error * 0.1875

    return image
