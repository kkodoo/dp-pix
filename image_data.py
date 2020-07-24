#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 11:25:22 2020

@author: aparnami
"""

import os
import random
import numpy as np
from PIL import Image

data_path = {
'omniglot': "/home/aparnami/Documents/GitHub/prototypical-noise/data/omniglot/data",
'miniimagenet': "/home/aparnami/Documents/GitHub/prototypical-noise/data/miniimagenet/data/train",
'faces':"/home/aparnami/Documents/GitHub/Facial-Similarity-with-Siamese-Networks-in-Pytorch/data/faces/training"        
}

def choose_random_path(path):
    files = os.listdir(path)
    file = random.choice(files)
    path = os.path.join(path, file)
    return path


class Dataset:
    def __init__(self):
        self.data_dir = ''
        self.scale = None
    
    def get_random_image():
        pass 
    
    def load_images(self, n=16):
         image_paths = set([self.get_random_image() for i in range(n)])
         images = list(map(Image.open, image_paths))
         return images


class OmniglotDataset(Dataset):
    def __init__(self):
        self.data_dir = data_path['omniglot']
        self.scale = (28,28)
    
    def get_random_image(self):
        alphabet = choose_random_path(self.data_dir)
        character = choose_random_path(alphabet)
        image = choose_random_path(character)
        return image
    
    def scale_up(self, image):
        np_image = np.asarray(image, dtype=np.float32)
        scaled = np_image * 255
        image = Image.fromarray(scaled.astype(np.uint8))
        return image
    
    def load_images(self, n=16):
        images = super().load_images(n)
        scaled = list(map(self.scale_up, images))
        return scaled
    
    
class MiniImageNetDataset(Dataset):
    def __init__(self):
        self.data_dir = data_path['miniimagenet']
        self.scale = (84,84)
    
    def get_random_image(self):
        label = choose_random_path(self.data_dir)
        image = choose_random_path(label)
        return image


class FacesDataset(Dataset):
    def __init__(self):
        self.data_dir = data_path['faces']
        self.scale = (92,112)
        
    def get_random_image(self):
        person = choose_random_path(self.data_dir)
        image = choose_random_path(person)
        return image