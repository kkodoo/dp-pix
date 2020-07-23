#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:42:15 2020

@author: aparnami
"""

import numpy as np
from PIL import Image

def add_laplace_noise(px_img, f_w, f_h, m=200, eps=0.01, noise_factor=0.5):
    """
    Input:
        px_img: pixelated image
        f_w, f_h: width and height of the window used for pixelation
        mean: mean for laplace noise
        eps: privacy parameter
        m: number of pixels to add noise to (see paper)
    Output:
        Return laplace pertubated image
    """
    # px_img = px_img / 255.0
    scale = (255 * m) / (f_w * f_h * eps)
    noise = noise_factor * np.random.laplace(loc=0,scale=scale, size=px_img.shape)
    noisy_image = np.clip(px_img + noise,0,255)
    return noisy_image


def pixelate(img, target_w, target_h, f_w, f_h):
    """
    Input: 
        img: numpy array for image
        target_w: required width of the pixelated image
        target_h: required height of the pixelated image
        f_w, f_h: width and height of the window used for pixelation
    Output:
        Returns pixelated image with dimenstion (target_h, target_w, input_channels)
    """
   
    
    assert img.shape[0] == target_h * f_h
    assert img.shape[1] == target_w * f_w
    
    flag = False
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
        flag = True
    
    ch = img.shape[2]
    
    px = np.zeros((target_h, target_w, ch))
    for i in range(target_h):
        row = i * f_h
        for j in range(target_w):
            col = j * f_w
            grid = img[row : row + f_h, col : col + f_w]
            m = np.mean(grid, axis=(0,1))
            px[i,j,:] = m
    
    if flag:
        px = np.squeeze(px)
    
    return px

def dp_pixelate(I, target_w, target_h, m, eps, noise_factor):
    """
    Input:
        I: Pillow image object
        target_w: required width of the pixelated image
        target_h: required height of the pixelated image
        eps: privacy parameter
        m: number of pixels to add noise to (see paper)
    Output:
        Return DP pixelated Pillow image with dimenstion (target_h, target_w, input_channels)
    """

    img = np.asarray(I, dtype=np.float32)
    shape = img.shape
    
    # Estimate the filter dimensions to get the desired shape
    f_h = shape[0] // (target_h )
    f_w = shape[1] // (target_w )
    
    # Area of the image that will be processed with given filter dimensions
    actual_h = target_h * f_h
    actual_w = target_w * f_w
    
    # Number of pixels that will be cropped in both dimensions
    crop_h = shape[0] - actual_h
    crop_w = shape[1] - actual_w
    
    # Image that will be left after evenly cropping both dimensions
    cropped_img = img[crop_h//2 : crop_h//2 + actual_h, crop_w//2 : crop_w//2 + actual_w]
    
    px_img = pixelate(cropped_img, target_w, target_h, f_w, f_h)
    dp_px_img = add_laplace_noise(px_img, f_w, f_h, m, eps, noise_factor)
    dp_px_I = Image.fromarray(dp_px_img.astype(np.uint8))
    
    return dp_px_I