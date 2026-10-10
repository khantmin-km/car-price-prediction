import dash
import dash_bootstrap_components as dbc
from dash import html

import config

# Create the app
app = dash.Dash(
    __name__,
    use_pages=True,       # to load all in pages/
    external_stylesheets=[dbc.themes.BOOTSTRAP],  # Bootstrap styling
    title="Car Price Predictor",  # Browser tab title
)

server = app.server


navbar = dbc.NavbarSimple(
    children=[
        # highlight the page that we are on
        dbc.NavLink(page["name"], href=page["relative_path"], active="exact")
        for page in dash.page_registry.values()
    ],
    brand="Car Price Predictor",
    color="dark",
    dark=True,
)


layout_children = [navbar]

# In demo mode, show a yellow banner so users know prices are not real
if config.DEMO_MODE:
    layout_children.append(
        dbc.Alert(
            "Demo mode: predictions shown are illustrative, not from a trained model.",
            color="warning",
            class_name="mb-0 text-center rounded-0",
        )
    )

# The current page's content goes here
layout_children.append(dbc.Container(dash.page_container, class_name="py-4"))

app.layout = html.Div(layout_children)

# Only runs when started directly with "python main.py"
if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
