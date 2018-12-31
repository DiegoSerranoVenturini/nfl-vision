from nflvision.utils.loader import ImgLoader
from nflvision.utils.plot import ImgPlotter
import numpy as np
from matplotlib import pyplot as plt


def test_load_img():

    PATH = "/Users/diegoserrano/Desktop/shotgun.jpeg"

    img = ImgLoader().load_img(PATH)

    assert isinstance(img, np.ndarray)


def test_load_images():

    img_loader = ImgLoader.build()

    for img in img_loader:
        print(img)
        print([img.shape for img in img])

    assert True


if __name__ == '__main__':

    test_load_images()
