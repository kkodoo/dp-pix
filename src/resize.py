import numpy as np

class Resize:

    @staticmethod
    def crop_image(img, target_h, target_w):
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
        return cropped_img, f_h, f_w


    @staticmethod
    def pad_image(img, target_h, target_w):
        shape = img.shape

        # pixels left out from each dimension
        extra_h = shape[0] % target_h
        extra_w = shape[1] % target_w

        # padding required so that dimensions are evenly divisible by target dimensions 
        pad_h = target_h - extra_h if extra_h != 0 else 0
        pad_w = target_w - extra_w if extra_w != 0 else 0

        # Evenly pad in both dimensions
        pad_h_before = pad_h // 2
        pad_h_after = pad_h - pad_h_before
        pad_w_before = pad_w // 2
        pad_w_after = pad_w - pad_w_before
        padded_img = np.pad(img, ((pad_h_before, pad_h_after),(pad_w_before, pad_w_after), (0,0)), mode='constant')

        new_shape = padded_img.shape
        f_h = new_shape[0] // target_h 
        f_w = new_shape[1] // target_w

        return padded_img, f_h, f_w