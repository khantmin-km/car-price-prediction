"""Dropdown options and input bounds for the car-detail forms.

Text choices (brand, fuel, etc.) are read from the saved model file. They must
match the training data exactly: the model drops the first category when it
one-hot encodes, so an unknown value would become all zeros, look the same as
that dropped category, and give a confident wrong price instead of an error.

Number limits are set to what a real car can be, not the range seen in the
training data, so an unusual but real car is never refused. The one exception
is YEAR_MIN, which is set where the model stops giving good predictions. 
"""

import datetime
import functools
import pathlib

import joblib

from src.preprocessing import OWNER_MAP

# Model path
MODEL_PATH = pathlib.Path(__file__).resolve().parent / "models" / "linear_model.joblib"

# This is the fallback categories in case no model is present so that our form still renders
_FALLBACK = {
    "fuel_options": ["Diesel", "Petrol"],
    "seller_type_options": ["Dealer", "Individual", "Trustmark Dealer"],
    "transmission_options": ["Automatic", "Manual"],
    "brand_options": [
        "Ambassador", "Ashok", "Audi", "BMW", "Chevrolet", "Daewoo", "Datsun",
        "Fiat", "Force", "Ford", "Honda", "Hyundai", "Isuzu", "Jaguar", "Jeep",
        "Kia", "Land", "Lexus", "MG", "Mahindra", "Maruti", "Mercedes-Benz",
        "Mitsubishi", "Nissan", "Opel", "Peugeot", "Renault", "Skoda", "Tata",
        "Toyota", "Volkswagen", "Volvo",
    ],
}


# Cache the result so the file is read from disk only once
@functools.lru_cache(maxsize=1)
def load_bundle() -> dict | None:
    """Load the trained model bundle, or None if it hasn't been produced yet."""
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def _option(key: str):
    # Use the saved list if the model file exists, otherwise the fallback list
    bundle = load_bundle()
    return bundle[key] if bundle is not None else _FALLBACK[key]


# Dropdown choices for each text feature
FUEL_OPTIONS = _option("fuel_options")
SELLER_TYPE_OPTIONS = _option("seller_type_options")
TRANSMISSION_OPTIONS = _option("transmission_options")
BRAND_OPTIONS = _option("brand_options")

# Test drive cars were removed from training, so we don't offer them as per instruction
OWNER_OPTIONS = [label for label in OWNER_MAP if label != "Test Drive Car"]

# Seat counts a passenger car can realistically have. Seats is a plain number
# So a count missing from the training data still works.
SEATS_OPTIONS = list(range(2, 15))

# Minimum year to accept (see limitation section in notebook)
YEAR_MIN = 2000

# We allow next year's model and the year is computed so that it is not stale.
YEAR_MAX = datetime.date.today().year + 1

# The threshold here comes from the cleaning step.
KM_DRIVEN_MIN, KM_DRIVEN_MAX = 0, 1_000_000

# We set limits based on plausible values for a car can have so that we reject stupid values.
MILEAGE_MIN, MILEAGE_MAX = 5.0, 50.0  # kmpl
ENGINE_MIN, ENGINE_MAX = 500.0, 8_000.0  # CC
MAX_POWER_MIN, MAX_POWER_MAX = 15.0, 1_000.0  # bhp
