import numpy as np


class Perceptron:

    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    """
    eta : float
    n_iter : int

    """ 
    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting. w[0] = threshold
    errors_ : list
        Number of miss classifications in each epoch.

    """
    w_ : []
    errors_ : list

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None  # defined in method fit

    def fit(self, X, y):

        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])  # First position corresponds to threshold
        self.errors = 0

        for _ in range(self.n_iter):
            errors = 0
            for i in zip(X, y):
                predict = self.predict(X)
                self.w_[1:] += self.eta * (y[i] - predict) * X[i]
                self.w_[0] += self.eta * (y[i] - predict)
            self.errors_.append(errors)

        return self

    def predict(self, X):
        """Return class label.
            First calculate the output: (X * weights) + threshold
            Second apply the step function
            Return a list with classes
        """

        sortida = np.dot(X, self.w_[1:]) + self.w_[0]
        return 1 if sortida > 0 else 0
