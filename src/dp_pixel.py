#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:42:15 2020

@author: aparnami
"""

import numpy as np
from .pixelate import Pixelate
from .noise import Noise
from .resize import Resize

def dp_pixelate(img, target_h, target_w, m, eps, 
                noise_factor = 1, 
                resize_f = Resize.pad_image, 
                pixelate_f = Pixelate.pytorch):
    """
    Input:
        img: numpy array of your image
        target_h: required height of the pixelated image
        target_w: required width of the pixelated image
        eps: privacy parameter
        m: number of pixels to add noise to (see paper)
        noise_factor: scale the noise by this factor (default: 1 i.e., don't scale).
        resize_f: Function to use in order to fit the target dimensions correctly. 
            Resize.pad_image: This function pads 0's at the image boundary. (default)
            Resize.crop_image: This function crops boundary pixels.
        pixelate_f: Function to use for pixelating the image. 
            All the methods below compute same result. They just differ in performance.
                Pixelate.sequential: Slowest
                Pixelate.skimage: Okay
                Pixelate.pytorch: Fastest (default)
    Output:
        Return DP pixelated image with dimension (target_h, target_w, input_channels)
    """

    flag = False
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
        flag = True
    
    num_channels = img.shape[2]
    resized_img, f_h, f_w = resize_f(img, target_h, target_w)
    px_img = pixelate_f(resized_img, f_h, f_w)
    
    # distributing eps among channels by eps/num_channels
    scale = (1 * m * num_channels) / (f_h * f_w * eps) 
    dp_px_img = np.zeros(px_img.shape)
    for i in range(num_channels):
        dp_px_img[:,:,i] = Noise.add_laplace_noise(px_img[:,:,i], 0, scale, noise_factor=noise_factor)
    
    if flag:
        dp_px_img = np.squeeze(dp_px_img)

    return dp_px_img