import numpy as np
from skimage.measure import block_reduce
import torch
from torch.nn.functional import avg_pool2d

class Pixelate:
    
    @staticmethod
    def sequential(img, f_h, f_w):
        target_h = img.shape[0] // f_h
        target_w = img.shape[1] // f_w 
        px = np.zeros((target_h, target_w, img.shape[2]))
        for i in range(target_h):
            row = i * f_h
            for j in range(target_w):
                col = j * f_w
                grid = img[row : row + f_h, col : col + f_w]
                m = np.mean(grid, axis=(0,1))
                px[i,j,:] = m
        return px

    @staticmethod
    def skimage(img, f_h, f_w):
        px = block_reduce(img, (f_h, f_w, 1), func=np.mean)
        return px

    @staticmethod
    def pytorch(img, f_h, f_w):
        img = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2)
        px = avg_pool2d(img, (f_h, f_w))
        px = px.permute(0,2,3,1).squeeze(0).numpy()
        return px