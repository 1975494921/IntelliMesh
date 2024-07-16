import numpy as np


# Adjusting MSELoss and SGD for NumPy support
class MSELoss:
    def __init__(self, predicted, target):
        self.predicted = predicted
        self.target = target

    def forward(self):
        # Compute the mean squared error for batches
        loss = ((self.predicted.data - self.target.data) ** 2).mean()
        return loss

    def backward(self):
        # Compute gradient of loss w.r.t predicted output for batches
        grad = 2 * (self.predicted.data - self.target.data) / self.predicted.data.size
        self.predicted.backward(grad)

    def __str__(self):
        return self.forward()

    def __repr__(self):
        return self.forward()

    def __format__(self, format_spec):
        return str(self.forward())


class CrossEntropyLoss:
    def __init__(self, predicted, target):
        self.predicted = predicted
        self.target = target

    def forward(self):
        # Number of samples in the batch
        batch_size = self.predicted.data.shape[0]

        # Compute log softmax
        # Log of exponential scores minus log sum of exp scores for numerical stability
        log_softmax = self.predicted.data - np.log(np.sum(np.exp(self.predicted.data), axis=1, keepdims=True))

        # Compute loss: Negative log likelihood
        # We use the target class indices to select the corresponding log softmax values
        loss = -np.sum(log_softmax[np.arange(batch_size), self.target.data]) / batch_size
        return loss

    def backward(self):
        # Number of samples
        batch_size = self.predicted.data.shape[0]

        # Gradient of log softmax
        grad = np.exp(self.predicted.data) / np.sum(np.exp(self.predicted.data), axis=1, keepdims=True)

        # Subtract 1 from the gradients where it matches the class index
        grad[np.arange(batch_size), self.target.data] -= 1

        # Average over the batch
        grad /= batch_size

        self.predicted.backward(grad)

    def __str__(self):
        return self.forward()

    def __repr__(self):
        return self.forward()

    def __format__(self, format_spec):
        return str(self.forward())
