"""Unit tests for the saved price-class model used by the app."""

import pathlib

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import apply_scaler, encode_for_model

MODEL_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "app" / "code" / "models" / "classifier_model.joblib"
)

# Saved model + training medians, column list, scaling numbers and cut points
bundle = joblib.load(MODEL_PATH)

# A few cars shaped like the app's form input; None = a field left blank
CARS = pd.DataFrame(
    [
        {"year": 2015, "km_driven": 60000, "brand": "Maruti", "fuel": "Petrol",
         "seller_type": "Individual", "transmission": "Manual", "owner": 1,
         "mileage": 21.0, "engine": 1197.0, "max_power": 82.0, "seats": 5},
        {"year": 2008, "km_driven": 150000, "brand": "Hyundai", "fuel": "Diesel",
         "seller_type": "Dealer", "transmission": "Manual", "owner": 2,
         "mileage": None, "engine": None, "max_power": None, "seats": 5},
        {"year": 2019, "km_driven": None, "brand": "BMW", "fuel": "Diesel",
         "seller_type": "Dealer", "transmission": "Automatic", "owner": 1,
         "mileage": 16.0, "engine": 1995.0, "max_power": 190.0, "seats": 5},
    ]
)


def prepare(cars):
    # Same filling, one-hot and scaling steps as training and the app
    encoded = encode_for_model(cars, bundle["impute_stats"], bundle["feature_columns"])
    return apply_scaler(encoded, bundle["scaler_stats"]).to_numpy(dtype=float)


def test_model_takes_expected_input():
    X = prepare(CARS)
    # One row per car, with the same columns the model was trained on
    assert X.shape == (len(CARS), len(bundle["feature_columns"]))
    # Blank fields were filled in, so no missing values reach the model
    assert not np.isnan(X).any()
    # The model accepts this input without an error
    bundle["model"].predict(X)


def test_output_has_expected_shape():
    pred = bundle["model"].predict(prepare(CARS))
    # One prediction per car
    assert pred.shape == (len(CARS),)
    # Each prediction is one of the 4 price classes
    assert set(pred.tolist()) <= {0, 1, 2, 3}
