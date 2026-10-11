"""The car-details form shared by every model page: the input fields, the
checks on what was typed, and turning the answers into a row for the model.

Each page passes its own id prefix (e.g. "lm"), because component ids must be
unique across the whole app.
"""

import dash_bootstrap_components as dbc
import pandas as pd
from dash import State, dcc, html

import options
from src.preprocessing import OWNER_MAP

# Form fields in the order the callbacks receive them: (id suffix, model column)
FIELDS = [
    ("year", "year"),
    ("km-driven", "km_driven"),
    ("brand", "brand"),
    ("fuel", "fuel"),
    ("seller-type", "seller_type"),
    ("transmission", "transmission"),
    ("owner", "owner"),
    ("mileage", "mileage"),
    ("engine", "engine"),
    ("max-power", "max_power"),
    ("seats", "seats"),
]


def _field(label, component, required=False):
    # Required fields get a red "*" next to the label
    label_children = [label, html.Span(" *", className="text-danger")] if required else label
    # Label + input in one column; 3 fields per row on medium+ screens
    return dbc.Col([dbc.Label(label_children), component], md=4, class_name="mb-3")


def _number_input(component_id, min_value, max_value, hint, step: str | int = "any"):
    """Numeric input bounded to the range the model accepts.

    The browser's own min/max give immediate feedback, but a value typed
    outside them still reaches the callback, so validate() re-checks it.
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


def form_fields(prefix: str) -> dbc.Row:
    """All the input fields, with ids like "<prefix>-year"."""
    # One input per model feature; ranges and choices come from options.py
    return dbc.Row(
        [
            _field(
                "Year",
                _number_input(
                    f"{prefix}-year",
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
                    f"{prefix}-km-driven",
                    options.KM_DRIVEN_MIN,
                    options.KM_DRIVEN_MAX,
                    f"{options.KM_DRIVEN_MIN:,}–{options.KM_DRIVEN_MAX:,}",
                    step=1,
                ),
            ),
            # Text features are dropdowns so only known categories are sent
            _field(
                "Brand",
                _dropdown(f"{prefix}-brand", options.BRAND_OPTIONS, placeholder="Select a brand"),
                required=True,
            ),
            _field(
                "Fuel type",
                _dropdown(f"{prefix}-fuel", options.FUEL_OPTIONS, placeholder="Select fuel type"),
                required=True,
            ),
            _field(
                "Seller type",
                _dropdown(
                    f"{prefix}-seller-type",
                    options.SELLER_TYPE_OPTIONS,
                    placeholder="Select seller type",
                ),
                required=True,
            ),
            _field(
                "Transmission",
                _dropdown(
                    f"{prefix}-transmission",
                    options.TRANSMISSION_OPTIONS,
                    placeholder="Select transmission",
                ),
                required=True,
            ),
            _field(
                "Ownership",
                _dropdown(f"{prefix}-owner", options.OWNER_OPTIONS, placeholder="Select ownership"),
                required=True,
            ),
            # Optional numeric fields (filled with the median if blank)
            _field(
                "Mileage (kmpl)",
                _number_input(
                    f"{prefix}-mileage",
                    options.MILEAGE_MIN,
                    options.MILEAGE_MAX,
                    f"{options.MILEAGE_MIN:g}–{options.MILEAGE_MAX:g}",
                ),
            ),
            _field(
                "Engine (CC)",
                _number_input(
                    f"{prefix}-engine",
                    options.ENGINE_MIN,
                    options.ENGINE_MAX,
                    f"{options.ENGINE_MIN:g}–{options.ENGINE_MAX:g}",
                ),
            ),
            _field(
                "Max power (bhp)",
                _number_input(
                    f"{prefix}-max-power",
                    options.MAX_POWER_MIN,
                    options.MAX_POWER_MAX,
                    f"{options.MAX_POWER_MIN:g}–{options.MAX_POWER_MAX:g}",
                ),
            ),
            _field(
                "Seats",
                _dropdown(f"{prefix}-seats", options.SEATS_OPTIONS, placeholder="Select seats"),
                required=True,
            ),
        ]
    )


def form_states(prefix: str) -> list[State]:
    """One State per field, in FIELDS order, for a page's callback."""
    # State reads the inputs without re-running on every keystroke
    return [State(f"{prefix}-{suffix}", "value") for suffix, _ in FIELDS]


def read_inputs(values) -> dict:
    """Map the callback's field values to the model's column names."""
    # Blank fields arrive as None and get median-imputed in encode_for_model()
    return {column: value for (_, column), value in zip(FIELDS, values)}


def validate(inputs: dict) -> list[str]:
    """Return one message per problem with the inputs; empty if all is fine."""
    # Fields that must be filled in before predicting
    required = [
        ("year", "Year"),
        ("brand", "Brand"),
        ("fuel", "Fuel type"),
        ("seller_type", "Seller type"),
        ("transmission", "Transmission"),
        ("owner", "Ownership"),
        ("seats", "Seats"),
    ]
    # One message for each empty required field
    errors = [f"{label} is required." for column, label in required if inputs[column] is None]

    year, km_driven = inputs["year"], inputs["km_driven"]
    mileage, engine, max_power = inputs["mileage"], inputs["engine"], inputs["max_power"]
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
    return errors


def error_box(errors: list[str]) -> dbc.Alert:
    # Show all problems at once in a red box instead of predicting
    return dbc.Alert([html.Div(e) for e in errors], color="danger")


def input_row(inputs: dict) -> pd.DataFrame:
    """One-row table from the inputs; owner text -> number, same as training."""
    return pd.DataFrame([{**inputs, "owner": OWNER_MAP[inputs["owner"]]}])
