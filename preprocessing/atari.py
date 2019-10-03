import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch


def imshow(image):
    # img = img / 2 + 0.5     # unnormalize
    plt.imshow(image, interpolation='nearest')
    plt.show()


def preprocess_breakout(image):
    image = image[32: 192]  # Crop the not necessary part of the image
    image = cv2.resize(image, (80, 80))
    image = image[::2, ::2, ::3]
    image[image != 0] = 1
    image = np.expand_dims(np.rollaxis(image, 2, 0), axis=0)
    return torch.from_numpy(image).type(torch.float)
