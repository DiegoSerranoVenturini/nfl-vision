import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader


class CnnBlock(nn.Module):

    def __init__(self, n_channels_in=3, n_channels_out=3, kernel_size=5, padding=2, stride=1, dilation=1, pool_kernel_size=None):

        super(CnnBlock, self).__init__()

        self.conv_kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.pool_kernel_size = pool_kernel_size

        self.cnn_block = nn.Sequential(
            nn.Conv2d(n_channels_in, n_channels_out, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(n_channels_out),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(pool_kernel_size) if pool_kernel_size else None

    def forward(self, features_t):

        out = self.cnn_block(features_t)

        if self.pool_kernel_size:

            out = self.pool(out)

        return out

    def output_size(self, input_size=256):

        if isinstance(input_size, tuple):
            w_size = input_size[0]
            h_size = input_size[1]
        else:
            w_size = h_size = input_size

        output_cnn_w = np.floor((w_size + 2*self.padding - self.dilation*(self.conv_kernel_size-1)-1) / self.stride + 1)
        output_cnn_h = np.floor((h_size + 2*self.padding - self.dilation*(self.conv_kernel_size-1)-1) / self.stride + 1)

        if self.pool_kernel_size:
            output_block_w = int(np.floor((output_cnn_w - (self.pool_kernel_size-1)-1) / self.pool_kernel_size + 1))
            output_block_h = int(np.floor((output_cnn_h - (self.pool_kernel_size-1)-1) / self.pool_kernel_size + 1))
        else:
            output_block_w = int(output_cnn_w)
            output_block_h = int(output_cnn_h)

        return (output_cnn_w, output_cnn_h), (output_block_w, output_block_h)


class CNN(nn.Module):

    def __init__(self, img_size, n_conv_blocks, n_classes, n_channels_in=3, channel_increase_rate=2, kernel_size=5, padding=2, stride=1,
                 dilation=1, pool_kernel_size=2):

        super(CNN, self).__init__()

        self.conv_blocks = nn.ModuleList(
            [CnnBlock(n_channels_in, n_channels_in*channel_increase_rate, kernel_size, padding, stride, dilation, pool_kernel_size)] +
            [CnnBlock(n_channels_in*(channel_increase_rate*i), n_channels_in*(i+1*channel_increase_rate), kernel_size, padding, stride, dilation, pool_kernel_size)
             for i in range(1, n_conv_blocks)]
        )

        img_size_output = img_size
        for cnn in self.conv_blocks:
            _, img_size_output = cnn.output_size(img_size_output)

        self.linear_out = nn.Linear(img_size_output[0] * img_size_output[1] * channel_increase_rate * n_conv_blocks * n_channels_in,
                                    n_classes)

    def forward(self, features_t):

        x = features_t

        for cnn in self.conv_blocks:
            x = cnn(x)

        x = x.view(x.size(0), -1)

        x = self.linear_out(x)
        out = F.log_softmax(x)

        return out
