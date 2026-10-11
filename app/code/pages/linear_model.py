import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, callback, html

import car_form
import config
import options
from src.preprocessing import encode_for_model

# Add this page to the app at /linear-model (also shows up in the nav bar)
dash.register_page(__name__, path="/linear-model", name="Linear Model", order=1)


# Page layout: title, short help text, input form, button and result area
layout = dbc.Container(
    [
        html.H1("Linear Model"),
        html.P(
            "Fill in what you know about the car below. Year, brand, fuel "
            "type, seller type, transmission, ownership, and seats are "
            "required; distance driven, mileage, engine size, and max power "
            "can be left blank and will be filled in automatically."
        ),
        # The shared car-details form, with ids starting "lm-"
        car_form.form_fields("lm"),
        # Clicking the button runs estimate_price() below
        dbc.Button("Estimate price", id="lm-submit", color="primary", class_name="mt-2"),
        # The price (or error messages) is shown here
        html.Div(id="lm-result", className="mt-4 fs-4"),
    ],
    class_name="py-4",
)


def predict_price(inputs: dict) -> float:
    """Return an estimated selling price for the given feature values."""
    # Demo mode returns a fixed price, so the app runs without a trained model
    if config.DEMO_MODE:
        return 450_000.0

    # Saved model + training medians + training column list
    bundle = options.load_bundle()
    row = car_form.input_row(inputs)
    # Same filling and one-hot steps as training, matched to training columns
    encoded = encode_for_model(row, bundle["impute_stats"], bundle["feature_columns"])
    # Model was trained on log(price), so its output is a log price
    log_price = bundle["model"].predict(encoded)[0]
    # We undo the log to get the price back in rupees
    return float(np.exp(log_price))


# Runs when the button is clicked
@callback(
    Output("lm-result", "children"),
    Input("lm-submit", "n_clicks"),
    *car_form.form_states("lm"),
    prevent_initial_call=True,  # Don't run when the page first loads
)
def estimate_price(n_clicks, *values):
    # Field values -> dict keyed by the model's column names
    inputs = car_form.read_inputs(values)
    errors = car_form.validate(inputs)
    if errors:
        return car_form.error_box(errors)

    price = predict_price(inputs)
    return f"Estimated selling price: ₹{price:,.0f}"
