import functools
import pathlib

import dash
import dash_bootstrap_components as dbc
import joblib
import numpy as np
from dash import Input, Output, callback, html

import car_form
import config
from src.preprocessing import apply_scaler, encode_for_model

# Add this page to the app at /custom-model (also shows up in the nav bar)
dash.register_page(__name__, path="/custom-model", name="Polynomial Model", order=2)

# Saved model trained in the notebook with our own gradient descent class
MODEL_PATH = pathlib.Path(__file__).resolve().parent.parent / "models" / "custom_model.joblib"

# Almost no training car is more powerful than this (see limitation section in notebook)
POWER_WARNING_BHP = 280


# Cache the result so the file is read from disk only once
@functools.lru_cache(maxsize=1)
def load_bundle() -> dict | None:
    """Load the trained model bundle, or None if it hasn't been produced yet."""
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def _comparison_text():
    """What is different about this model, with the scores saved alongside it."""
    bundle = load_bundle()
    intro = (
        "This model can follow curves in the data, for example extra power raising "
        "the price more for some cars than for others, where the Linear Model can "
        "only draw straight lines."
    )
    # Without the saved file (e.g. demo mode) we can't quote the scores
    if bundle is None:
        return intro
    scores = bundle["scores"]
    return (
        f"{intro} Tested on the same cars, it is more accurate for typical cars: its "
        f"usual error is {scores['same_folds_median_error_pct']:.1f}% of the price, "
        f"against {scores['old_same_folds_median_error_pct']:.1f}% for the Linear Model. "
        f"It is less reliable for very powerful cars (above about {POWER_WARNING_BHP} bhp), "
        "because there were almost none of them to learn from."
    )


# Page layout: title, how to use it, what is new, the form, button and result area
layout = dbc.Container(
    [
        html.H1("Polynomial Model"),
        html.P(
            "Fill in what you know about the car below. Year, brand, fuel "
            "type, seller type, transmission, ownership, and seats are "
            "required; distance driven, mileage, engine size, and max power "
            "can be left blank and will be filled in automatically."
        ),
        dbc.Alert([html.Strong("What's different: "), _comparison_text()], color="info"),
        # The shared car-details form, with ids starting "cm-"
        car_form.form_fields("cm"),
        # Clicking the button runs estimate_price() below
        dbc.Button("Estimate price", id="cm-submit", color="primary", class_name="mt-2"),
        # The price (or error messages) is shown here
        html.Div(id="cm-result", className="mt-4 fs-4"),
    ],
    class_name="py-4",
)


def predict_price(inputs: dict) -> float:
    """Return an estimated selling price for the given feature values."""
    # Demo mode returns a fixed price, so the app runs without a trained model
    if config.DEMO_MODE:
        return 450_000.0

    # Saved model + training medians, column list and scaling numbers
    bundle = load_bundle()
    row = car_form.input_row(inputs)
    # Same filling and one-hot steps as training, matched to training columns
    encoded = encode_for_model(row, bundle["impute_stats"], bundle["feature_columns"])
    # Same scaling as training; this also adds the degree-2 columns
    prepared = apply_scaler(encoded, bundle["scaler_stats"])
    # Model was trained on log(price), so its output is a log price
    log_price = bundle["model"].predict(prepared)[0]
    # We undo the log to get the price back in rupees
    return float(np.exp(log_price))


# Runs when the button is clicked
@callback(
    Output("cm-result", "children"),
    Input("cm-submit", "n_clicks"),
    *car_form.form_states("cm"),
    prevent_initial_call=True,  # Don't run when the page first loads
)
def estimate_price(n_clicks, *values):
    # Field values -> dict keyed by the model's column names
    inputs = car_form.read_inputs(values)
    errors = car_form.validate(inputs)
    if errors:
        return car_form.error_box(errors)

    price = predict_price(inputs)
    result = [html.Div(f"Estimated selling price: ₹{price:,.0f}")]
    # Still give a price for a very powerful car, but say it is a rough guess
    if inputs["max_power"] is not None and inputs["max_power"] > POWER_WARNING_BHP:
        result.append(
            dbc.Alert(
                f"Very few cars in the training data have more than {POWER_WARNING_BHP} bhp, "
                "so this price is a rough guess.",
                color="warning",
                class_name="mt-3 fs-6",
            )
        )
    return result
