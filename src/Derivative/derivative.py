import numpy as np


class Derivative(object):
    def __init__(self, f, h=0.001, order=1, n_points=3, vectorized=False):
        self.f = np.vectorize(f) if vectorized else f
        self.h = h
        self.order = order
        self.n_points = n_points
        self.weights = self._weights()

    def _weights(self):
        if self.order == 1:
            if self.n_points == 3:
                return [0.5, -3, 4, -1]
            if self.n_points == 5:
                return [1 / 12, -25, 48, -36, 16, -3]

    def __str__(self):
        return "Difference Method"

    def description(self):
        pass

    def __call__(self, x0):
        fp = 0
        for i, w in enumerate(self.weights):
            if i == 0:
                continue
            fp += w * self.f(x0 + (i - 1) * self.h)
        fp *= self.weights[0] / self.h
        return fp
