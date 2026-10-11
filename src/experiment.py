"""Cross-validation and MLflow logging for the gradient-descent regression models."""

import time
from itertools import product
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.entities import Metric
from sklearn.model_selection import train_test_split

from src.linear_regression import Lasso, LinearRegression, Normal, Ridge
from src.preprocessing import apply_scaler, encode_for_model, fit_impute_stats, fit_scaler

# The four model types we compare
MODEL_TYPES = ("normal", "ridge", "lasso", "polynomial")
# Only these two use the penalty strength lam
PENALIZED = ("ridge", "lasso")
# MLflow accepts at most 1000 metric values per batch call
_BATCH_LIMIT = 1000
# Scores logged on each setting's parent run
SUMMARY_KEYS = ("cv_r2_mean", "cv_r2_std", "cv_mse_mean", "cv_mse_std", "cv_r2_log_mean",
                "diverged_folds", "mean_epochs", "fit_seconds")


def setup_mlflow(experiment_name: str, root) -> None:
    """Log to a SQLite file `mlflow.db` in `root`, saving any files (artifacts)
    to `root/mlartifacts`.
    """
    root = Path(root).resolve()
    mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
    # The artifact folder can only be chosen when the experiment is first created
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=(root / "mlartifacts").as_uri())
    mlflow.set_experiment(experiment_name)


def build_model(model_type: str, lam: float, **params) -> LinearRegression:
    """Make an untrained model of the given type. Polynomial is a plain model
    trained on extra degree-2 columns, so it gets no penalty.
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {model_type!r}")
    if model_type == "ridge":
        return Ridge(lam, **params)
    if model_type == "lasso":
        return Lasso(lam, **params)
    return Normal(**params)


def prepare_features(X_fit: pd.DataFrame, others: list, feature_columns: list[str], polynomial: bool):
    """Fill missing values, one-hot encode and scale.

    Medians, means and standard deviations are learned from `X_fit` only, then
    applied to `X_fit` and to every frame in `others`. Returns the list of
    prepared frames (X_fit first) and the learned stats.
    """
    impute_stats = fit_impute_stats(X_fit)
    # The fixed column list keeps the same columns even if X_fit lacks a rare brand
    fit_encoded = encode_for_model(X_fit, impute_stats, feature_columns)
    scaler_stats = fit_scaler(fit_encoded, polynomial=polynomial)

    prepared = [apply_scaler(fit_encoded, scaler_stats)]
    for X in others:
        prepared.append(apply_scaler(encode_for_model(X, impute_stats, feature_columns), scaler_stats))
    stats = {"impute_stats": impute_stats, "scaler_stats": scaler_stats, "feature_columns": feature_columns}
    return prepared, stats


def fit_with_holdout(model: LinearRegression, X, y, holdout_size: float = 0.1, seed: int = 42):
    """Train on most of X, y and keep `holdout_size` of the rows aside only to
    decide when to stop early.
    """
    X_fit, X_stop, y_fit, y_stop = train_test_split(X, y, test_size=holdout_size, random_state=seed)
    return model.fit(X_fit, y_fit, X_stop, y_stop)


def score_rupees(y_true_log, y_pred_log) -> dict:
    """R^2 and MSE on the real price in rupees, plus R^2 on the log price."""
    y_true_log = np.asarray(y_true_log, dtype=float)
    # Turn log prices back into rupees 
    # A wild prediction becomes inf instead of a warning
    with np.errstate(over="ignore"):
        y_true, y_pred = np.exp(y_true_log), np.exp(y_pred_log)
    return {
        "r2": LinearRegression.r2(y_true, y_pred),
        "mse": LinearRegression.mse(y_true, y_pred),
        "r2_log": LinearRegression.r2(y_true_log, y_pred_log),
    }


def _log_loss_curves(run_id: str, model: LinearRegression) -> None:
    """Log the loss after every epoch, sending many values per call to stay fast."""
    now = int(time.time() * 1000)
    metrics = [Metric("train_loss", v, now, i) for i, v in enumerate(model.train_losses_)]
    metrics += [Metric("holdout_loss", v, now, i) for i, v in enumerate(model.val_losses_)]
    client = mlflow.MlflowClient()
    for start in range(0, len(metrics), _BATCH_LIMIT):
        client.log_batch(run_id, metrics=metrics[start : start + _BATCH_LIMIT])


def cross_validate(model_type: str, X, y_log, feature_columns, cv, lam: float = 0.0,
                   holdout_size: float = 0.1, seed: int = 42, **params) -> list[dict]:
    """Cross-validate one setting. Each fold is logged as its own nested MLflow run.

    X is the cleaned (not yet encoded) training data and y_log is log(price).
    Returns one dict of scores per fold.
    """
    polynomial = model_type == "polynomial"
    y_log = pd.Series(np.asarray(y_log, dtype=float), index=X.index)
    folds = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]
        # Everything is learned from this fold's training rows only
        (X_tr, X_val), _ = prepare_features(X_tr, [X_val], feature_columns, polynomial)

        with mlflow.start_run(run_name=f"fold-{fold}", nested=True) as run:
            model = build_model(model_type, lam, random_state=seed, **params)
            start = time.perf_counter()
            fit_with_holdout(model, X_tr, y_tr, holdout_size, seed)
            seconds = time.perf_counter() - start

            result = {"fold": fold, "diverged": model.diverged_, "epochs_run": model.epochs_run_,
                      "best_epoch": model.best_epoch_, "fit_seconds": seconds}
            # A run that blew up has no usable weights, so it gets no scores
            if model.diverged_:
                result.update(r2=np.nan, mse=np.nan, r2_log=np.nan)
            else:
                result.update(score_rupees(y_val, model.predict(X_val)))

            mlflow.log_param("fold", fold)
            # MLflow stores numbers only, so we turn True/False and None into numbers
            mlflow.log_metrics({k: float(np.nan if v is None else v) for k, v in result.items() if k != "fold"})
            _log_loss_curves(run.info.run_id, model)
        folds.append(result)
    return folds


def run_name(setting: dict) -> str:
    """Short readable name for one setting, e.g. 'ridge · mini · xavier · lr=0.001 · mom=0.9'."""
    return (f"{setting['model']} · {setting['method']} · {setting['init']}"
            f" · lr={setting['lr']} · mom={setting['momentum']}")


def _finished_runs() -> pd.DataFrame:
    """Finished setting-level runs in the current experiment. Fold runs are left
    out because they have no 'seed' param.
    """
    runs = mlflow.search_runs(filter_string="attributes.status = 'FINISHED'")
    if runs.empty or "params.seed" not in runs:
        return pd.DataFrame()
    return runs[runs["params.seed"].notna()]


def _saved_summary(done: pd.DataFrame, name: str, logged: dict) -> dict | None:
    """Scores of an earlier finished run with this name and exactly these params, if any."""
    if done.empty:
        return None
    match = done[done["tags.mlflow.runName"] == name]
    # MLflow stores params as text, so compare them as text
    for key, value in logged.items():
        column = f"params.{key}"
        if column not in match:
            return None
        match = match[match[column] == str(value)]
    if match.empty:
        return None
    # search_runs lists the newest run first
    run = match.iloc[0]
    summary = {k: float(run[f"metrics.{k}"]) for k in SUMMARY_KEYS}
    summary["diverged_folds"] = int(summary["diverged_folds"])
    return summary


def run_grid(grid: dict, X, y_log, feature_columns, cv, lam: float = 0.0,
             holdout_size: float = 0.1, seed: int = 42, skip_existing: bool = True,
             **fixed) -> pd.DataFrame:
    """Cross-validate every combination of the values in `grid` and log each one
    as an MLflow run, with its folds nested inside.

    `grid` maps "model", "method", "init", "lr" and "momentum" to lists of
    values. `fixed` holds settings shared by every run (num_epochs, batch_size,
    patience, tol). With `skip_existing`, a setting already finished in MLflow
    with the same name and params is read back instead of trained again, so an
    interrupted grid picks up where it stopped. Returns one row per setting.
    """
    keys = list(grid)
    settings = [dict(zip(keys, values)) for values in product(*grid.values())]
    done = _finished_runs() if skip_existing else pd.DataFrame()
    rows = []

    for i, setting in enumerate(settings, start=1):
        model_type = setting["model"]
        params = {k: v for k, v in setting.items() if k != "model"} | fixed
        name = run_name(setting)
        # Every value that defines this setting
        # Also used to find it again later
        logged = setting | fixed | {
            "lam": lam if model_type in PENALIZED else 0.0,
            "folds": cv.get_n_splits(), "holdout_size": holdout_size, "seed": seed,
        }

        summary = _saved_summary(done, name, logged)
        if summary is not None:
            rows.append(setting | summary)
            print(f"[{i}/{len(settings)}] {name}: already in MLflow, reused")
            continue

        with mlflow.start_run(run_name=name):
            # Record every setting, so runs can be filtered and compared in the MLflow UI
            mlflow.log_params(logged)
            folds = cross_validate(model_type, X, y_log, feature_columns, cv, lam, holdout_size, seed, **params)
            f = pd.DataFrame(folds)

            # If any fold blew up, the setting as a whole counts as failed
            diverged = int(f["diverged"].sum())
            summary = {
                "cv_r2_mean": np.nan if diverged else f["r2"].mean(),
                "cv_r2_std": np.nan if diverged else f["r2"].std(ddof=0),
                "cv_mse_mean": np.nan if diverged else f["mse"].mean(),
                "cv_mse_std": np.nan if diverged else f["mse"].std(ddof=0),
                "cv_r2_log_mean": np.nan if diverged else f["r2_log"].mean(),
                "diverged_folds": diverged,
                "mean_epochs": f["epochs_run"].mean(),
                "fit_seconds": f["fit_seconds"].sum(),
            }
            mlflow.log_metrics({k: float(v) for k, v in summary.items()})

        rows.append(setting | summary)
        # One progress line per setting, since the full grid takes a while
        r2 = "diverged" if diverged else f"R2 {summary['cv_r2_mean']:.4f}"
        print(f"[{i}/{len(settings)}] {name}: {r2}, {summary['mean_epochs']:.0f} epochs, "
              f"{summary['fit_seconds']:.1f}s")

    return pd.DataFrame(rows)
