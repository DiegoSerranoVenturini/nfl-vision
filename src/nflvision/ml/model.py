import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import balanced_accuracy_score
import logging

from nflvision.cfg.constants import ImageFormatConstants

log = logging.getLogger(__name__)


class ImgClassifier:

    estimator = None
    _loss_fn = None
    _optimizer = None

    def build(self, net: nn.Module, optimizer, loss_fn):

        self.estimator = net
        self._loss_fn = loss_fn
        self._optimizer = optimizer

        return self

    def fit(self, train_loader, num_epochs=1, init_lr=0.01):
        log.info("Estimator fit.")

        losses = []
        for epoch in range(num_epochs):
            start = time.time()
            # log.info("Epoch {}/{}.".format(epoch, num_epochs))
            print("Epoch {}/{}".format(epoch+1, num_epochs))

            optimizer = self._optimizer(params=self.estimator.parameters(), lr=init_lr)

            for images, labels in train_loader:

                images = Variable(images.float())
                labels = Variable(labels.long())

                # forward pass
                loss, optimizer = self._forward_pass(images, labels, optimizer)

                # backward pass
                loss.backward()

                # calculate the gradients
                optimizer.step()

                # log the losses
                losses.append(loss.item())

            elapsed = time.time() - start
            log.info("Epoch {}/{}. Seconds elpased: {}".format(epoch, num_epochs, round(elapsed, 1)))

            # print('Epoch : %d/%d, Loss: %.4f' % (epoch+1, num_epochs, losses))
        return self

    def _forward_pass(self, features, label, optimizer):

        # restart the gradient calculations
        optimizer.zero_grad()

        # Forward pass
        outputs = self.estimator(features)

        # calculate the loss function
        criterion = self._loss_fn()
        loss = criterion(outputs, label)

        return loss, optimizer

    def predict(self, features):

        # TODO: change this to be not a constant but a configuration argument
        if not isinstance(features, torch.Tensor):

            img_size = ImageFormatConstants.IMG_SIZE

            features = torch.Tensor(features.reshape(-1, 1, img_size, img_size))

        predictions = np.argmax(F.log_softmax(self.estimator(features)).data.numpy(), axis=1)

        return predictions


class ImgClassifierEvaluator:

    @staticmethod
    def evaluate(classifier: ImgClassifier, evaluation_loader):
        log.info("Evaluating estimator results.")

        accuracies = []

        for batch_images, batch_labels in evaluation_loader:

            y_pred = classifier.predict(batch_images)
            y_true = batch_labels.data.numpy()

            # todo: the metric should be a parameter or configuration file
            batch_acc = balanced_accuracy_score(y_true, y_pred)

            accuracies.append(batch_acc)

        return np.mean(accuracies)








