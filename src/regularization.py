"""Penalty terms added to the loss to keep model weights small (regularization).

Each penalty is called with the model's full parameter array `theta`, whose
first row is the intercept. The intercept is never penalized.
"""

import numpy as np


def _without_intercept(theta: np.ndarray) -> np.ndarray:
    """Copy of `theta` with the intercept (row 0) set to zero."""
    # Copy so the model's own theta is not changed
    w = np.array(theta, dtype=float)
    # The intercept only shifts every prediction up or down, so we leave it free
    w[0] = 0.0
    return w


class NoPenalty:
    """No penalty, for plain (normal) and polynomial regression."""

    def __call__(self, theta: np.ndarray) -> float:
        # Adds nothing to the loss
        return 0.0

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        # Adds nothing to the gradient
        return np.zeros_like(theta, dtype=float)


class LassoPenalty:
    """L1 penalty: lam * sum(|w|). Can push weak weights all the way to zero."""

    def __init__(self, lam: float):
        # Lambda: how strongly large weights are punished
        self.lam = lam

    def __call__(self, theta: np.ndarray) -> float:
        # __call__ lets us use the object like a function: penalty(theta)
        return self.lam * np.sum(np.abs(_without_intercept(theta)))

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        # Slope of |w| is +1 for positive w, -1 for negative w, 0 at zero
        return self.lam * np.sign(_without_intercept(theta))


class RidgePenalty:
    """L2 penalty: lam * sum(w^2). Shrinks all weights but rarely to exactly zero."""

    def __init__(self, lam: float):
        # Lambda: how strongly large weights are punished
        self.lam = lam

    def __call__(self, theta: np.ndarray) -> float:
        return self.lam * np.sum(np.square(_without_intercept(theta)))

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        # Slope of w^2 is 2w
        return self.lam * 2 * _without_intercept(theta)
