import matplotlib.pyplot as plt


def imshow(image):
    # img = img / 2 + 0.5     # unnormalize
    plt.imshow(image, interpolation='nearest')
    plt.show()


def preprocess_breakout(image):
    image = image[30: 196, 6: 153]  # Crop the not necessary part of the image
    image = image[::2, ::2, 0]
    image[image != 0] = 1
    return image
