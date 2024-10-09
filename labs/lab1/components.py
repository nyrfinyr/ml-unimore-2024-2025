"""
Class that models a Naive Bayes Classifier
"""

import numpy as np


class NonParametricNaiveBayesClassifier:
    """
    Naive Bayes Classifier.
    Training:
    For each class, a naive likelihood model is estimated for P(X/Y),
    and the prior probability P(Y) is computed.
    Inference:
    performed according with the Bayes rule:
    P = argmax_Y (P(X/Y) * P(Y))
    or
    P = argmax_Y (log(P(X/Y)) + log(P(Y)))
    """

    def __init__(self):
        self._classes = None
        self._n_classes = 0
        self._eps = np.finfo(np.float32).eps
        self._class_priors = np.zeros(self._n_classes)
        self._pixel_probs_given_class = []

    def fit(self, X, Y):
        # Compute priors
        self._classes, counts = np.unique(Y, return_counts=True)
        self._class_priors = counts / len(Y)
        self._n_classes = len(self._classes)

        _, h, w = X.shape
        self._pixel_probs_given_class = np.zeros((self._n_classes, h, w), dtype=np.float64)

        # Compute likelihoods as histograms of probabilities
        for c in range(self._n_classes):
            X_c = X[Y == c]  ### find all the samples labeled with class c
            self._pixel_probs_given_class[c] = np.sum(X_c, axis=0) / len(Y)  ### Compute likelihood for class c

    def predict(self, X):
        account_for_zeros = lambda p_i: 1 - p_i
        predictions = []
        for x_i in X:  # for each sample
            all_posteriors = []
            for c in self._classes:  # for each model
                X_i = self._pixel_probs_given_class[c]
                posterior_probability = (np.sum(np.log(X_i[x_i == 1]))
                                         + np.sum(np.log(account_for_zeros(X_i[x_i == 0])))
                                         + np.log(self._class_priors[c]))
                all_posteriors.append(posterior_probability)
            predictions.append(np.argmax(all_posteriors))
        return np.array(predictions)
