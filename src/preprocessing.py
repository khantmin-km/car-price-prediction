"""Cleaning, feature-encoding and scaling utilities for the car price dataset."""

from itertools import combinations_with_replacement

import numpy as np
import pandas as pd

# Turn owner text into numbers so the order is kept
OWNER_MAP = {
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4,
    "Test Drive Car": 5,
}


def _first_token_to_float(series: pd.Series) -> pd.Series:
    """Take the leading numeric token of a "<value> <unit>" string column.
    Handles blanks (already NaN in pandas) and the odd malformed value
    by coercing failures to NaN rather than raising, since a handful of rows
    in this dataset have those.
    """
    # "23.4 kmpl" -> "23.4" (keep only the part before the first space)
    first_token = series.str.split().str[0]
    # Convert to a number; anything that can't be converted becomes NaN
    return pd.to_numeric(first_token, errors="coerce")


def clean_car_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw car price dataset: normalize categorical fields, parse
    numeric values out of unit-suffixed strings, drop unusable rows/columns,
    and remove exact duplicate rows.
    Returns a new dataframe.
    """
    # Work on a copy so the original dataframe is not changed
    out = df.copy()

    # Remove rows that are exact copies of another row
    out = out.drop_duplicates()

    # Torque column is dropped per instruction
    out = out.drop(columns=["torque"])

    # CNG/LPG mileage are removed per instruction
    out = out[~out["fuel"].isin(["CNG", "LPG"])].copy()

    # Replace owner text with numbers (1-5)
    out["owner"] = out["owner"].map(OWNER_MAP)
    # Test drive cars are removed per instruction
    out = out[out["owner"] != OWNER_MAP["Test Drive Car"]].copy()

    # Strip units ("kmpl", "CC", "bhp") and keep just the numbers
    out["mileage"] = _first_token_to_float(out["mileage"])
    out["engine"] = _first_token_to_float(out["engine"])
    out["max_power"] = _first_token_to_float(out["max_power"])

    # A value of 0 is not real here, so we treat it as missing
    out.loc[out["mileage"] == 0, "mileage"] = np.nan
    out.loc[out["max_power"] == 0, "max_power"] = np.nan

    # Over 1 million km is almost surely a typo (see notebook for judgement)
    huge_km = out["km_driven"] > 1_000_000
    # Median km of the same car model, ignoring the typo rows themselves
    name_median_km = out["km_driven"].where(~huge_km).groupby(out["name"]).transform("median")
    # Replace each typo with that model's median
    out.loc[huge_km, "km_driven"] = name_median_km[huge_km]

    # Car brand is the first word of the name (e.g. "Maruti Swift" -> "Maruti")
    out["brand"] = out["name"].str.split().str[0]
    # Full name has too many unique values to be useful, so drop it
    out = out.drop(columns=["name"])

    # Renumber rows 0, 1, 2... after the rows we removed
    return out.reset_index(drop=True)


# Numeric columns that may have missing values 
NUMERIC_IMPUTE_COLS = ["year", "km_driven", "mileage", "engine", "max_power", "seats"]
# Text columns that get turned into 0/1 columns
CATEGORICAL_COLS = ["fuel", "seller_type", "transmission", "brand"]


def fit_impute_stats(df_clean: pd.DataFrame) -> dict:
    """Compute the median of each numeric feature that can have missing values."""
    # Should be called on training data only, so test data stays unseen
    return {col: float(df_clean[col].median()) for col in NUMERIC_IMPUTE_COLS}


def encode_for_model(
    df: pd.DataFrame, impute_stats: dict, feature_columns: list[str] | None = None
) -> pd.DataFrame:
    """Turn a cleaned dataframe (or a single new row) into a model-ready feature
    matrix: fill missing numeric values, one-hot encode categorical columns,
    and align the result to a fixed set of feature columns.

    `feature_columns` should be the exact column set produced by encoding the
    training data, so a differently-shaped input (e.g. a row missing some
    categories) still produces a matrix with the same columns, in the same
    order, that the model was trained on. Omit it only to discover that
    column set in the first place, by encoding the training data itself.
    """
    # Always work on a copy
    out = df.copy()

    # Fill missing numbers with the training medians
    for col in NUMERIC_IMPUTE_COLS:
        # Skip columns that aren't in this input
        if col in out.columns:
            out[col] = out[col].fillna(impute_stats[col]).astype(float)

    # Only encode the text columns that are actually present
    dummy_cols = [c for c in CATEGORICAL_COLS if c in out.columns]
    # Make one hot column per category and drop the first one on training data
    out = pd.get_dummies(out, columns=dummy_cols, drop_first=(feature_columns is None))

    # Training data: return as-is, its columns become the reference list
    if feature_columns is None:
        return out
    return out.reindex(columns=feature_columns, fill_value=0)


# Every column that holds a number rather than a 0/1 category flag
NUMERIC_COLS = ["year", "km_driven", "owner", "mileage", "engine", "max_power", "seats"]


def add_polynomial_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Add every degree-2 term of `cols`: each column squared (`year^2`) and
    each pair multiplied together (`year*engine`).
    """
    new_cols = {}
    # Every pair of columns, including a column paired with itself
    for a, b in combinations_with_replacement(cols, 2):
        name = f"{a}^2" if a == b else f"{a}*{b}"
        new_cols[name] = df[a] * df[b]
    # Add all new columns at once, after the existing ones
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def _column_stats(df: pd.DataFrame, cols: list[str]) -> dict:
    """Mean and standard deviation of each column."""
    # A column with one constant value has std 0; use 1 so we never divide by zero
    return {col: (float(df[col].mean()), float(df[col].std(ddof=0)) or 1.0) for col in cols}


def _scale(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Turn each column in `stats` into (value - mean) / std."""
    out = df.copy()
    for col, (mean, std) in stats.items():
        out[col] = (out[col] - mean) / std
    return out


def fit_scaler(df: pd.DataFrame, polynomial: bool = False) -> dict:
    """Learn the mean and standard deviation of each numeric column, and of the
    degree-2 columns too when `polynomial` is True. 0/1 columns are not scaled.
    """
    # Should be called on training data only, so test data stays unseen
    numeric = _column_stats(df, NUMERIC_COLS)
    poly = {}
    if polynomial:
        # Square the scaled values, not the raw ones: raw year and year^2 move
        # almost exactly together, so the model could not tell them apart
        expanded = add_polynomial_features(_scale(df, numeric), NUMERIC_COLS)
        new_cols = [c for c in expanded.columns if c not in df.columns]
        # The new columns get their own mean and std so they are on the same scale
        poly = _column_stats(expanded, new_cols)
    return {"polynomial": polynomial, "numeric": numeric, "poly": poly}


def apply_scaler(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Scale an encoded dataframe with stats from `fit_scaler`, adding the
    degree-2 columns first if the stats were fit with `polynomial=True`.
    """
    out = _scale(df, stats["numeric"])
    if stats["polynomial"]:
        # Same steps, same order as in fit_scaler
        out = add_polynomial_features(out, NUMERIC_COLS)
        out = _scale(out, stats["poly"])
    return out
