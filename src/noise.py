import numpy as np

class Noise:

    @staticmethod
    def add_gaussian_noise(img, mean, stdev, noise_factor=1):
        img = img / 255.0
        noise = noise_factor * np.random.normal(mean, stdev, img.shape)
        noisy_image = np.clip(img + noise, 0, 1) * 255
        return noisy_image

    @staticmethod
    def add_laplace_noise(img, loc, scale, noise_factor=1):
        img = img / 255.0
        noise = noise_factor *  np.random.laplace(loc, scale, img.shape)
        noisy_image = np.clip(img + noise, 0, 1) * 255
        return noisy_image