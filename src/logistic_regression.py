"""Multinomial logistic regression trained with gradient descent, written from scratch.

The model learns W, one column of weights per class; row 0 of W is the intercept.
Callers pass only the feature columns; the column of 1s is added inside the class.
The class also computes the classification metrics (accuracy, precision, recall, f1).
"""

import numpy as np
import pandas as pd

from src.regularization import NoPenalty

# Tiny number added inside log so we never take log(0)
EPS = 1e-12


class LogisticRegression:
    """Softmax (multinomial) logistic regression.

    k: number of classes; labels must be 0, 1, ..., k-1.
    penalty: RidgePenalty(lam) for ridge, or None for no penalty.
    method: "batch" (all rows per step), "mini" (batch_size rows per step) or
        "sto" (one row per step).
    max_iter: number of update steps.
    batch_size: rows per step for "mini"; None means 30% of the rows.
    """

    def __init__(self, k=4, penalty=None, method="batch", lr=0.001, max_iter=5000,
                 batch_size=None, random_state=None):
        self.k = k
        self.penalty = penalty if penalty is not None else NoPenalty()
        self.method = method
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state

    # ----- training -----

    def fit(self, X, y):
        X = self._add_intercept(X)
        Y = self._one_hot(y)
        m, n = X.shape
        rng = np.random.default_rng(self.random_state)
        self.W = rng.random((n, self.k))
        self.losses = []

        if self.method == "batch":
            for _ in range(self.max_iter):
                # Every step uses all rows
                self._step(X, Y)

        elif self.method == "mini":
            batch_size = self.batch_size or int(0.3 * m)
            for _ in range(self.max_iter):
                # Random start, kept far enough from the end that the batch is always full
                ix = rng.integers(0, m - batch_size + 1)
                self._step(X[ix:ix + batch_size], Y[ix:ix + batch_size])

        elif self.method == "sto":
            used = set()
            for _ in range(self.max_iter):
                # Pick a row not used yet in this pass over the data
                idx = rng.integers(m)
                while idx in used:
                    idx = rng.integers(m)
                used.add(idx)
                self._step(X[idx:idx + 1], Y[idx:idx + 1])
                # Every row used once: start a new pass
                if len(used) == m:
                    used = set()

        else:
            raise ValueError('method must be "batch", "mini" or "sto"')
        return self

    def _step(self, X, Y):
        """One gradient descent update on the rows given."""
        loss, grad = self.gradient(X, Y)
        self.losses.append(loss)
        self.W = self.W - self.lr * grad

    def gradient(self, X, Y):
        """Loss and gradient on these rows. The cross-entropy part is averaged over
        the rows so one learning rate works for every method. the penalty is added on top.
        """
        m = X.shape[0]
        h = self.h_theta(X, self.W)
        loss = -np.sum(Y * np.log(h + EPS)) / m + self.penalty(self.W)
        error = h - Y
        grad = X.T @ error / m + self.penalty.gradient(self.W)
        return loss, grad

    def softmax(self, theta_t_x):
        # Subtract each row's max first so exp() can't overflow; the result is unchanged
        z = np.exp(theta_t_x - theta_t_x.max(axis=1, keepdims=True))
        # Each row becomes probabilities that add up to 1
        return z / z.sum(axis=1, keepdims=True)

    def h_theta(self, X, W):
        """Class probabilities, shape (m, k)."""
        return self.softmax(X @ W)

    def predict(self, X):
        # The class with the highest probability
        return np.argmax(self.h_theta(self._add_intercept(X), self.W), axis=1)

    def _one_hot(self, y):
        # Labels 0..k-1 -> rows with a single 1 in the column of the true class
        return np.eye(self.k)[np.asarray(y, dtype=int)]

    @staticmethod
    def _add_intercept(X):
        # Column of 1s in front, so row 0 of W acts as the intercept
        X = np.asarray(X, dtype=float)
        return np.hstack([np.ones((X.shape[0], 1)), X])

    # ----- metrics -----

    def _counts(self, y_true, y_pred):
        """True positives, false positives and false negatives for each class."""
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        classes = np.arange(self.k)[:, None]
        tp = np.sum((y_pred == classes) & (y_true == classes), axis=1)
        fp = np.sum((y_pred == classes) & (y_true != classes), axis=1)
        fn = np.sum((y_pred != classes) & (y_true == classes), axis=1)
        return tp, fp, fn

    @staticmethod
    def _safe_divide(a, b):
        # Gives 0 where b is 0 (a class never predicted or never present), like sklearn
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    def _support(self, y_true):
        # Number of true rows in each class
        return np.bincount(np.asarray(y_true, dtype=int), minlength=self.k)

    def accuracy(self, y_true, y_pred):
        # Correct predictions / all predictions
        return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

    def precision(self, y_true, y_pred):
        # Per class: TP / (TP + FP)
        tp, fp, _ = self._counts(y_true, y_pred)
        return self._safe_divide(tp, tp + fp)

    def recall(self, y_true, y_pred):
        # Per class: TP / (TP + FN)
        tp, _, fn = self._counts(y_true, y_pred)
        return self._safe_divide(tp, tp + fn)

    def f1(self, y_true, y_pred):
        # Per class: 2 * precision * recall / (precision + recall)
        p, r = self.precision(y_true, y_pred), self.recall(y_true, y_pred)
        return self._safe_divide(2 * p * r, p + r)

    def macro_precision(self, y_true, y_pred):
        # Plain average over the classes
        return float(np.mean(self.precision(y_true, y_pred)))

    def macro_recall(self, y_true, y_pred):
        return float(np.mean(self.recall(y_true, y_pred)))

    def macro_f1(self, y_true, y_pred):
        return float(np.mean(self.f1(y_true, y_pred)))

    def _weighted(self, scores, y_true):
        # Each class weighted by its share of the true labels; the weights add up to 1,
        # so no extra division by the number of classes (same as sklearn)
        weights = self._support(y_true) / len(y_true)
        return float(np.sum(weights * scores))

    def weighted_precision(self, y_true, y_pred):
        return self._weighted(self.precision(y_true, y_pred), y_true)

    def weighted_recall(self, y_true, y_pred):
        return self._weighted(self.recall(y_true, y_pred), y_true)

    def weighted_f1(self, y_true, y_pred):
        return self._weighted(self.f1(y_true, y_pred), y_true)

    def classification_report(self, y_true, y_pred):
        """Table laid out like sklearn's classification_report, built from the methods above."""
        support = self._support(y_true)
        report = pd.DataFrame({
            "precision": self.precision(y_true, y_pred),
            "recall": self.recall(y_true, y_pred),
            "f1-score": self.f1(y_true, y_pred),
            "support": support,
        }, index=[str(c) for c in range(self.k)])
        total = int(support.sum())
        # Accuracy row: sklearn shows only the score and the total count
        report.loc["accuracy"] = [np.nan, np.nan, self.accuracy(y_true, y_pred), total]
        report.loc["macro avg"] = [self.macro_precision(y_true, y_pred),
                                   self.macro_recall(y_true, y_pred),
                                   self.macro_f1(y_true, y_pred), total]
        report.loc["weighted avg"] = [self.weighted_precision(y_true, y_pred),
                                      self.weighted_recall(y_true, y_pred),
                                      self.weighted_f1(y_true, y_pred), total]
        report["support"] = report["support"].astype(int)
        return report
