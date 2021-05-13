import numpy as np

from Derivative import derivative


class Newton(object):
    def __init__(self, f, init, tol=1e-5, N=20):
        self.f = f
        self.init = init
        self.tol = tol
        self.N = N

    def __str__(self):
        return "Newton's Method"

    def description(self):
        return "To find a solution to f(x) = 0 given an initial approximation p0"

    def __call__(self):
        i = 1
        p0 = self.init
        fp = derivative.Derivative(self.f)
        while i < self.N:
            p = p0 - self.f(p0) / fp(p0)
            if np.abs(p - p0) < self.tol:
                return p
            i += 1
            p0 = p
        return f"The method failed after {self.N} iterations"
