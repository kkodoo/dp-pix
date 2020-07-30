from PIL import Image
from resize import Resize
from pixelate import Pixelate
import image_util as iu
import numpy as np

class Scale:

    @staticmethod
    def with_pillow(img, target_h, target_w):
        img = np.squeeze(img)
        I = iu.numpy_to_pillow(img)
        I_scaled = I.resize((target_w, target_h))
        scaled = iu.pillow_to_numpy(I_scaled)
        return scaled

    @staticmethod
    def crop_and_pixelate(img, target_h, target_w):
        img, f_h, f_w = Resize.crop_image(img, target_h, target_w)
        scaled = Pixelate.pytorch(img, f_h, f_w)
        return scaled
    
    @staticmethod
    def pad_and_pixelate(img, target_h, target_w):
        img, f_h, f_w = Resize.pad_image(img, target_h, target_w)
        scaled = Pixelate.pytorch(img, f_h, f_w)
        return scaled
