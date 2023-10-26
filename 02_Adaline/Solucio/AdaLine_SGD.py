import numpy as np
from numpy.random import seed

class Adaline(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.
    shuffle : bool (default: True)
        Shuffles training dat every epoch if True to prevent cycles.
    random_state : int (default: None)
        Set random state for shuffling and initializing the weights.

    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.cost_ = None

        if random_state:
            seed(random_state)

    def fit(self, X, y):
        """ Fit training dat.

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
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):

            if self.shuffle:
                X, y = self.__shuffle(X, y)

            cost = []
            for xi, target in zip(X, y):

                error = (target - self.net_output(xi))
                self.w_[1:] += self.eta * xi.dot(error)
                self.w_[0] += self.eta * error

                cost_parcial = 0.5 * error ** 2
                cost.append(cost_parcial)

            # Calcul del cost
            avg_cost = sum(cost) / len(y)
            cost = (avg_cost ** 2).sum() / 2.0
            self.cost_.append(cost)

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.net_input(xi)
        error = (target - output)

        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

    def __shuffle(self, X, y):
        """Shuffle training dat"""
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def net_output(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_output(X) >= 0.0, 1, -1)
