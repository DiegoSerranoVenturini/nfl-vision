from matplotlib import pyplot as plt
import numpy as np
import torch


class ImgPlotter:

    @classmethod
    def plot(cls, img):

        if isinstance(img, np.ndarray):
            cls._plot_from_numpy(img)

        elif isinstance(img, torch.Tensor):
            cls._plot_from_tensor(img)

    @staticmethod
    def _plot_from_numpy(img):

        plt.imshow(img)
        plt.show()

    @classmethod
    def _plot_from_tensor(cls, img: torch.Tensor):

        cls._plot_from_numpy(img.numpy())

