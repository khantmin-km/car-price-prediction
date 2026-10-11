"""Linear regression trained with gradient descent, written from scratch.

The model learns theta = [intercept, w1, ..., wn]. Callers pass only the
feature columns; the column of 1s for the intercept is added inside the class.
"""

import numpy as np

from src.regularization import LassoPenalty, NoPenalty, RidgePenalty

# Ways to split the data into update steps
METHODS = ("batch", "mini", "sto")
# Ways to set the starting weights
INITS = ("zeros", "xavier")


class LinearRegression:
    """Linear regression fitted by gradient descent.

    method: "batch" (all rows per step), "mini" (batch_size rows per step) or
        "sto" (one row per step). One epoch is always one full pass over the rows.
    init: "zeros" or "xavier" starting weights.
    momentum: share of the previous step carried into the next one, in [0, 1).
        0 means no momentum.
    patience / tol: with a validation set, stop after `patience` epochs without
        the validation loss improving by more than `tol`, and keep the best weights.
    """

    def __init__(
        self,
        penalty=None,
        lr: float = 0.001,
        method: str = "batch",
        batch_size: int = 50,
        num_epochs: int = 500,
        init: str = "zeros",
        momentum: float = 0.0,
        patience: int = 10,
        tol: float = 1e-5,
        random_state: int | None = None,
    ):
        # Catch typos early instead of training the wrong thing silently
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if init not in INITS:
            raise ValueError(f"init must be one of {INITS}, got {init!r}")
        if not 0 <= momentum < 1:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")

        # No penalty given means plain linear regression
        self.penalty = penalty if penalty is not None else NoPenalty()
        self.lr = lr
        self.method = method
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.init = init
        self.momentum = momentum
        self.patience = patience
        self.tol = tol
        self.random_state = random_state

    # ----- metrics -----

    @staticmethod
    def mse(y_true, y_pred) -> float:
        """Mean squared error: the average of (prediction - actual)^2."""
        y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
        return float(np.mean((y_pred - y_true) ** 2))

    @staticmethod
    def r2(y_true, y_pred) -> float:
        """R^2 score: 1 - (model's squared error) / (squared error of always guessing the mean)."""
        y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        return float(1 - ss_res / ss_tot)

    # ----- training -----

    def fit(self, X, y, X_val=None, y_val=None):
        """Train on X, y. If X_val and y_val are given, stop early when the
        validation loss stops improving and keep the best weights seen.
        """
        X = self._add_intercept(self._to_array(X))
        y = np.asarray(y, dtype=float).ravel()
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val = self._add_intercept(self._to_array(X_val))
            y_val = np.asarray(y_val, dtype=float).ravel()

        # One random generator for the starting weights and the shuffling, so a run can be repeated
        rng = np.random.default_rng(self.random_state)
        self.n_features_ = X.shape[1] - 1
        self.theta = self._init_theta(self.n_features_, rng)
        # Previous step for momentum; nothing to carry over at the start
        self.prev_step = np.zeros_like(self.theta)

        self.train_losses_, self.val_losses_ = [], []
        self.diverged_ = False
        # Loss before any training; a working run only ever gets below this
        start_loss = self.mse(y, X @ self.theta)
        best_val, best_theta, bad_epochs = np.inf, self.theta.copy(), 0
        self.best_epoch_ = None
        self.epochs_run_ = 0

        # Overflow is handled by the divergence check below, so hide numpy's warnings
        with np.errstate(over="ignore", invalid="ignore"):
            for epoch in range(self.num_epochs):
                self._run_epoch(X, y, rng)
                self.epochs_run_ = epoch + 1

                # Loss on the whole training set, so every method is measured the same way
                train_loss = self.mse(y, X @ self.theta)
                # Steps too big: the loss climbed above where it started (or hit inf/NaN),
                # so the weights are blowing up; give up on this run
                if not np.isfinite(train_loss) or train_loss > start_loss:
                    self.diverged_ = True
                    break
                self.train_losses_.append(train_loss)

                if has_val:
                    val_loss = self.mse(y_val, X_val @ self.theta)
                    self.val_losses_.append(val_loss)
                    # Count it as better only if it improved by more than tol
                    if val_loss < best_val - self.tol:
                        best_val, best_theta, bad_epochs = val_loss, self.theta.copy(), 0
                        self.best_epoch_ = epoch
                    else:
                        bad_epochs += 1
                        # No real progress for `patience` epochs in a row: stop
                        if bad_epochs >= self.patience:
                            break

        # Go back to the weights from the best validation epoch
        if has_val and not self.diverged_:
            self.theta = best_theta
        return self

    def _init_theta(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Starting weights. The intercept always starts at 0."""
        theta = np.zeros(n + 1)
        if self.init == "xavier":
            # Random values spread evenly between -1/sqrt(n) and 1/sqrt(n), n = number of features
            limit = 1.0 / np.sqrt(n)
            theta[1:] = rng.uniform(-limit, limit, size=n)
        return theta

    def _run_epoch(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> None:
        """One full pass over the training rows."""
        if self.method == "batch":
            # Batch: one step using every row
            self._update(X, y)
            return
        # Shuffle so the batches differ every epoch
        perm = rng.permutation(X.shape[0])
        X, y = X[perm], y[perm]
        # Stochastic is mini-batch with a batch of one row
        size = 1 if self.method == "sto" else self.batch_size
        for start in range(0, X.shape[0], size):
            self._update(X[start : start + size], y[start : start + size])

    def _update(self, X: np.ndarray, y: np.ndarray) -> None:
        """One gradient descent step on the rows given."""
        m = X.shape[0]
        # Slope of the loss (1/2m) * sum((X @ theta - y)^2), plus the penalty's slope
        grad = (1 / m) * X.T @ (X @ self.theta - y) + self.penalty.gradient(self.theta)
        # Momentum: add part of the previous step to this one
        step = self.lr * grad + self.momentum * self.prev_step
        # Subtract the whole step, so the carried-over part also moves downhill
        self.theta = self.theta - step
        self.prev_step = step

    # ----- prediction -----

    def predict(self, X) -> np.ndarray:
        """Predict for each row of X (same columns, same order as in training)."""
        X = self._to_array(X)
        # The weights only make sense for the columns the model was trained on
        if X.shape[1] != self.n_features_:
            raise ValueError(f"expected {self.n_features_} features, got {X.shape[1]}")
        return self._add_intercept(X) @ self.theta

    @staticmethod
    def _to_array(X) -> np.ndarray:
        """Turn a DataFrame or array into a 2-D float array (0/1 bool columns become 0.0/1.0)."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (rows x features), got {X.ndim}-D")
        return X

    @staticmethod
    def _add_intercept(X: np.ndarray) -> np.ndarray:
        """Put a column of 1s in front, so theta[0] works as the intercept."""
        return np.hstack([np.ones((X.shape[0], 1)), X])

    @property
    def coef_(self) -> np.ndarray:
        """Feature weights, without the intercept."""
        return self.theta[1:]

    @property
    def intercept_(self) -> float:
        """The intercept (prediction when every feature is 0)."""
        return float(self.theta[0])

    # ----- feature importance -----

    def plot_feature_importance(self, feature_names, top_n: int | None = None, ax=None):
        """Horizontal bar chart of the coefficients, biggest size at the top.

        Comparing sizes only makes sense when the features share a scale, so
        numeric features should be standardized before fitting.
        """
        # Imported here so training and prediction never need matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        if len(feature_names) != self.n_features_:
            raise ValueError(f"expected {self.n_features_} names, got {len(feature_names)}")

        coef = self.coef_
        # Order by size, ignoring the sign, biggest first
        order = np.argsort(np.abs(coef))[::-1]
        if top_n is not None:
            order = order[:top_n]
        # barh draws from the bottom up, so reverse to put the biggest on top
        order = order[::-1]
        names = [feature_names[i] for i in order]
        values = coef[order]

        # Blue pushes the prediction up, red pushes it down
        up, down = "#2a78d6", "#e34948"
        colors = [up if v > 0 else down for v in values]

        if ax is None:
            _, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(order))))
        ax.barh(names, values, color=colors, height=0.6)
        ax.axvline(0, color="#555555", linewidth=0.8)
        ax.set_xlabel("Coefficient")
        ax.set_title("Feature importance (coefficients)")
        # Light grid and no box, so the bars stand out
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(
            handles=[Patch(color=up, label="Raises prediction"), Patch(color=down, label="Lowers prediction")],
            loc="lower right",
            frameon=False,
        )
        return ax


class Normal(LinearRegression):
    """Plain linear regression, no penalty."""

    def __init__(self, **kwargs):
        super().__init__(penalty=NoPenalty(), **kwargs)


class Ridge(LinearRegression):
    """Linear regression with an L2 penalty of strength `lam`."""

    def __init__(self, lam: float, **kwargs):
        super().__init__(penalty=RidgePenalty(lam), **kwargs)


class Lasso(LinearRegression):
    """Linear regression with an L1 penalty of strength `lam`."""

    def __init__(self, lam: float, **kwargs):
        super().__init__(penalty=LassoPenalty(lam), **kwargs)
