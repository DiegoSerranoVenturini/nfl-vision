import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ImgModelBuilder:

    _net = None
    _loss_fn = None
    _optimizer = None

    def build(self, net: nn.Module, optimizer, loss_fn):

        self._net = net
        self._loss_fn = loss_fn
        self._optimizer = optimizer

        return self

    def fit(self, train_loader, num_epochs=1, init_lr=0.01):

        losses = []
        for epoch in range(num_epochs):

            self._optimizer = self._optimizer(params=self._net.parameters(), lr=init_lr)

            for images, labels in train_loader:

                images = Variable(images.float())
                labels = Variable(labels.long())

                # forward pass
                loss = self._forward_pass(images, labels)

                # backward pass
                loss.backward()

                # calculate the gradients
                self._optimizer.step()

                # log the losses
                losses.append(loss.item())

            # print('Epoch : %d/%d, Loss: %.4f' % (epoch+1, num_epochs, losses))
        return self

    def _forward_pass(self, features, label):

        # restart the gradient calculations
        self._optimizer.zero_grad()

        # Forward pass
        outputs = self._net(features)

        # calculate the loss function
        criterion = self._loss_fn()
        loss = criterion(outputs, label)

        return loss





