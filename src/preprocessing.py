"""Cleaning and feature-encoding utilities for the car price dataset."""

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
