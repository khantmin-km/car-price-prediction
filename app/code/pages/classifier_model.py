import functools
import pathlib

import dash
import dash_bootstrap_components as dbc
import joblib
from dash import Input, Output, callback, html

import car_form
import config
from src.preprocessing import apply_scaler, encode_for_model

# Add this page to the app at /classifier-model (also shows up in the nav bar)
dash.register_page(__name__, path="/classifier-model", name="Price Class", order=3)

# Saved model trained in the notebook with our own logistic regression class
MODEL_PATH = pathlib.Path(__file__).resolve().parent.parent / "models" / "classifier_model.joblib"


# Cache the result so the file is read from disk only once
@functools.lru_cache(maxsize=1)
def load_bundle() -> dict:
    return joblib.load(MODEL_PATH)


# Page layout: title, how to use it, the form, button and result area
layout = dbc.Container(
    [
        html.H1("Price Class"),
        html.P(
            "This model puts the car into one of four price ranges instead of "
            "guessing an exact price. Year, brand, fuel type, seller type, "
            "transmission, ownership, and seats are required; distance driven, "
            "mileage, engine size, and max power can be left blank."
        ),
        # The shared car-details form, with ids starting "cl-"
        car_form.form_fields("cl"),
        # Clicking the button runs show_class() below
        dbc.Button("Predict price range", id="cl-submit", color="primary", class_name="mt-2"),
        # The price range (or error messages) is shown here
        html.Div(id="cl-result", className="mt-4 fs-4"),
    ],
    class_name="py-4",
)


def range_text(price_class: int, edges: list) -> str:
    """Price range of a class; a car exactly on a cut point is in the higher class."""
    if price_class == 0:
        return f"below ₹{edges[0]:,.0f}"
    if price_class == len(edges):
        return f"₹{edges[-1]:,.0f} and up"
    return f"₹{edges[price_class - 1]:,.0f} to under ₹{edges[price_class]:,.0f}"


def predict_range(inputs: dict) -> str:
    """Return the predicted price class and its range as text."""
    # Demo mode gives a fixed answer, so the app runs without a trained model
    if config.DEMO_MODE:
        return "Class 1: ₹250,000 to under ₹415,000"

    # Saved model + training medians, column list, scaling numbers and cut points
    bundle = load_bundle()
    row = car_form.input_row(inputs)
    # Same filling, one-hot and scaling steps as training
    encoded = encode_for_model(row, bundle["impute_stats"], bundle["feature_columns"])
    prepared = apply_scaler(encoded, bundle["scaler_stats"]).to_numpy(dtype=float)
    price_class = int(bundle["model"].predict(prepared)[0])
    return f"Class {price_class}: {range_text(price_class, bundle['price_edges'])}"


# Runs when the button is clicked
@callback(
    Output("cl-result", "children"),
    Input("cl-submit", "n_clicks"),
    *car_form.form_states("cl"),
    prevent_initial_call=True,  # Don't run when the page first loads
)
def show_class(n_clicks, *values):
    # Field values -> dict keyed by the model's column names
    inputs = car_form.read_inputs(values)
    errors = car_form.validate(inputs)
    if errors:
        return car_form.error_box(errors)
    return f"Predicted price range: {predict_range(inputs)}"
