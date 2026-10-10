import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dcc, html

import config
import options
from src.preprocessing import OWNER_MAP, encode_for_model

# Add this page to the app at /linear-model (also shows up in the nav bar)
dash.register_page(__name__, path="/linear-model", name="Linear Model")


def _field(label, component, required=False):
    # Required fields get a red "*" next to the label
    label_children = [label, html.Span(" *", className="text-danger")] if required else label
    # Label + input in one column; 3 fields per row on medium+ screens
    return dbc.Col([dbc.Label(label_children), component], md=4, class_name="mb-3")


def _number_input(component_id, min_value, max_value, hint, step: str | int = "any"):
    """Numeric input bounded to the range the model accepts.

    The browser's own min/max give immediate feedback, but a value typed
    outside them still reaches the callback, so estimate_price() re-checks it.
    """
    return dbc.Input(
        id=component_id,
        type="number",
        min=min_value,
        max=max_value,
        step=step,  # "any" allows decimals, 1 allows whole numbers only
        placeholder=hint,  # Shows the allowed range inside the empty box
    )


def _dropdown(component_id, dropdown_options, placeholder="Select an option"):
    return dcc.Dropdown(
        id=component_id,
        # Each choice is shown as text and sent to the callback as-is
        options=[{"label": str(o), "value": o} for o in dropdown_options],
        placeholder=placeholder,
        clearable=True,  # User can clear the choice with an "x"
        searchable=True,  # User can type to filter the list
    )


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
        # One input per model feature; ranges and choices come from options.py
        dbc.Row(
            [
                _field(
                    "Year",
                    _number_input(
                        "lm-year",
                        options.YEAR_MIN,
                        options.YEAR_MAX,
                        f"{options.YEAR_MIN}–{options.YEAR_MAX}",
                        step=1,
                    ),
                    required=True,
                ),
                # Optional: left blank -> we filled with the training median
                _field(
                    "Distance driven (km)",
                    _number_input(
                        "lm-km-driven",
                        options.KM_DRIVEN_MIN,
                        options.KM_DRIVEN_MAX,
                        f"{options.KM_DRIVEN_MIN:,}–{options.KM_DRIVEN_MAX:,}",
                        step=1,
                    ),
                ),
                # Text features are dropdowns so only known categories are sent
                _field(
                    "Brand",
                    _dropdown("lm-brand", options.BRAND_OPTIONS, placeholder="Select a brand"),
                    required=True,
                ),
                _field(
                    "Fuel type",
                    _dropdown("lm-fuel", options.FUEL_OPTIONS, placeholder="Select fuel type"),
                    required=True,
                ),
                _field(
                    "Seller type",
                    _dropdown(
                        "lm-seller-type",
                        options.SELLER_TYPE_OPTIONS,
                        placeholder="Select seller type",
                    ),
                    required=True,
                ),
                _field(
                    "Transmission",
                    _dropdown(
                        "lm-transmission",
                        options.TRANSMISSION_OPTIONS,
                        placeholder="Select transmission",
                    ),
                    required=True,
                ),
                _field(
                    "Ownership",
                    _dropdown("lm-owner", options.OWNER_OPTIONS, placeholder="Select ownership"),
                    required=True,
                ),
                # Optional numeric fields (filled with the median if blank)
                _field(
                    "Mileage (kmpl)",
                    _number_input(
                        "lm-mileage",
                        options.MILEAGE_MIN,
                        options.MILEAGE_MAX,
                        f"{options.MILEAGE_MIN:g}–{options.MILEAGE_MAX:g}",
                    ),
                ),
                _field(
                    "Engine (CC)",
                    _number_input(
                        "lm-engine",
                        options.ENGINE_MIN,
                        options.ENGINE_MAX,
                        f"{options.ENGINE_MIN:g}–{options.ENGINE_MAX:g}",
                    ),
                ),
                _field(
                    "Max power (bhp)",
                    _number_input(
                        "lm-max-power",
                        options.MAX_POWER_MIN,
                        options.MAX_POWER_MAX,
                        f"{options.MAX_POWER_MIN:g}–{options.MAX_POWER_MAX:g}",
                    ),
                ),
                _field(
                    "Seats",
                    _dropdown("lm-seats", options.SEATS_OPTIONS, placeholder="Select seats"),
                    required=True,
                ),
            ]
        ),
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
    # One-row table from the inputs; owner text -> number, same as training
    row = pd.DataFrame([{**inputs, "owner": OWNER_MAP[inputs["owner"]]}])
    # Same filling and one-hot steps as training, matched to training columns
    encoded = encode_for_model(row, bundle["impute_stats"], bundle["feature_columns"])
    # Model was trained on log(price), so its output is a log price
    log_price = bundle["model"].predict(encoded)[0]
    # We undo the log to get the price back in rupees
    return float(np.exp(log_price))


# Runs when the button is clicked; State reads the inputs without
# re-running on every keystroke
@callback(
    Output("lm-result", "children"),
    Input("lm-submit", "n_clicks"),
    State("lm-year", "value"),
    State("lm-km-driven", "value"),
    State("lm-brand", "value"),
    State("lm-fuel", "value"),
    State("lm-seller-type", "value"),
    State("lm-transmission", "value"),
    State("lm-owner", "value"),
    State("lm-mileage", "value"),
    State("lm-engine", "value"),
    State("lm-max-power", "value"),
    State("lm-seats", "value"),
    prevent_initial_call=True,  # Don't run when the page first loads
)
def estimate_price(
    n_clicks, year, km_driven, brand, fuel, seller_type, transmission, owner,
    mileage, engine, max_power, seats,
):
    # Fields that must be filled in before predicting
    required = [
        (year, "Year"),
        (brand, "Brand"),
        (fuel, "Fuel type"),
        (seller_type, "Seller type"),
        (transmission, "Transmission"),
        (owner, "Ownership"),
        (seats, "Seats"),
    ]
    # One message for each empty required field
    errors = [f"{label} is required." for value, label in required if value is None]

    # We check ranges again since the browser still lets out-of-range values through
    if year is not None and not (options.YEAR_MIN <= year <= options.YEAR_MAX):
        errors.append(f"Year must be between {options.YEAR_MIN} and {options.YEAR_MAX}.")
    if km_driven is not None and not (options.KM_DRIVEN_MIN <= km_driven <= options.KM_DRIVEN_MAX):
        errors.append(
            f"Distance driven must be between {options.KM_DRIVEN_MIN:,} and "
            f"{options.KM_DRIVEN_MAX:,} km."
        )
    if mileage is not None and not (options.MILEAGE_MIN <= mileage <= options.MILEAGE_MAX):
        errors.append(
            f"Mileage must be between {options.MILEAGE_MIN:g} and {options.MILEAGE_MAX:g} kmpl."
        )
    if engine is not None and not (options.ENGINE_MIN <= engine <= options.ENGINE_MAX):
        errors.append(
            f"Engine size must be between {options.ENGINE_MIN:g} and {options.ENGINE_MAX:g} CC."
        )
    if max_power is not None and not (options.MAX_POWER_MIN <= max_power <= options.MAX_POWER_MAX):
        errors.append(
            f"Max power must be between {options.MAX_POWER_MIN:g} and {options.MAX_POWER_MAX:g} bhp."
        )

    # Show all problems at once in a red box instead of predicting
    if errors:
        return dbc.Alert([html.Div(e) for e in errors], color="danger")

    # Gather inputs with the same column names the model was trained on
    inputs = {
        "year": year,
        "km_driven": km_driven,
        "brand": brand,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        # Blank fields arrive as None and get median-imputed in encode_for_model();
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
    }
    price = predict_price(inputs)
    return f"Estimated selling price: ₹{price:,.0f}"
