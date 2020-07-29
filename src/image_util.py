#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 16:40:52 2020

@author: aparnami
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def pillow_to_numpy(I):
    img = np.asarray(I, dtype=np.float32)
    return img

def numpy_to_pillow(img):
    I = Image.fromarray(img.astype(np.uint8))
    return I



def display_image_grid(images, size=(12,12), titles=None, num_cols=4):
    images = list(map(numpy_to_pillow, images))
    fig = plt.figure(figsize=size)
    fig.tight_layout(pad=0)
    N = len(images)
    cols = num_cols
    rows = N/cols if N%cols == 0 else (N//cols + 1)
    for i in range(N):
        ax = fig.add_subplot(rows, cols, i+1)
        plt.imshow(images[i], cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        if titles is not None:
            ax.set_title(titles[i])