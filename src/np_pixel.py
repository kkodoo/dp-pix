import numpy as np
from scale import Scale
from noise import Noise

def np_pixelate(img, target_h, target_w, sigma, 
                noise_factor = 1, 
                scale_f = Scale.pad_and_pixelate):
    """
    Input:
        img: numpy array of your image
        target_h: required height of the pixelated image
        target_w: required width of the pixelated image
        sigma: standard deviation for gaussian noise
        noise_factor: scale the noise by this factor (default: 1 i.e., don't scale).
        scale_f: function for scaling the image to target_h, target_w
            Scale.with_pillow: Uses pillow image library. Scaled image is interpolated.
            Scale.crop_and_pixelate: crops boundary pixels and then performs pixelation.
            Scale.pad_and_pixelate: pads 0's to boundary pixels and then performs pixelation. (default) 
    Output:
        Return: non private pixelated image with gaussian noise of dimension (target_h, target_w, input_channels)
    """

    flag = False
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
        flag = True

   
    scaled_img = scale_f(img, target_h, target_w)
    np_px_img = Noise.add_gaussian_noise(scaled_img, 0, sigma, noise_factor=noise_factor)
   
    if flag:
        np_px_img = np.squeeze(np_px_img)

    return np_px_img