import numpy as np

eps = np.finfo(float).eps

# https://en.wikipedia.org/wiki/Sigmoid_function
def sigmoid(x):
    """
    Element-wise sigmoid function
    """
    logistic = lambda x_i: 1 / (1 + np.exp(-x_i))
    return logistic(np.squeeze(x))


# https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy
def loss(y_true_list, y_pred_list):
    """
    The binary crossentropy loss.
    """
    cross_entropy_loss = 0
    for y_ground, y_pred in zip(y_true_list, y_pred_list):
        cross_entropy_loss += -(y_ground * np.log(y_pred + eps) + (1 - y_ground) * np.log(1 - y_pred + eps))
    return cross_entropy_loss


# − (X @ (Y − F(X, w))) / N
def dloss_dw(y_true, y_pred, X):
    """
    Derivative of loss function w.r.t. weights.
    """
    N = X.shape[0]
    return - X.T @ np.expand_dims(y_true - y_pred, axis=1) / N


class LogisticRegression:
    """ Models a logistic regression classifier. """

    def __init__(self):

        # weights placeholder
        self._w = None

    def fit_gd(self, X, Y, n_epochs, learning_rate, verbose=False):
        """
        Implements the gradient descent training procedure.
        """
        _, n_features = X.shape
        # weight initialization
        self._w = np.random.randn(n_features) * 0.001
        self._w = np.expand_dims(self._w, axis=1)

        for e in range(n_epochs):
            p = sigmoid(X @ self._w)

            Loss = loss(Y, p)

            if verbose and e % 500 == 0:
                print(f'Epoch {e:4d}: loss={Loss}')

            self._w = self._w - learning_rate * dloss_dw(Y, p, X)

    def predict(self, X):
        """
        Function that predicts.
        """

        p = sigmoid(X @ self._w)
        return np.where(p > 0.5, 1, 0)
