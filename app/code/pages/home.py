import dash
import dash_bootstrap_components as dbc
from dash import html

dash.register_page(__name__, path="/", name="Home")

# Our title and description here
layout = dbc.Container(
    [
        html.H1("Car Price Predictor"),
        html.P(
            "Estimate a used car's selling price from its year, mileage, fuel type, "
            "and a handful of other details. Pick a model from the menu above and "
            "fill in the car's details — each page lists which fields are "
            "required and which can be left blank."
        ),
    ],
    class_name="py-4",  
)
