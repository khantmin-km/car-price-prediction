# Car Price Prediction

A web application that estimates the selling price of a used car from its year, distance driven, brand, fuel type, and a handful of other details. The estimate comes from a linear regression model trained on 6,827 cleaned listings.

## What it does

The navigation bar lists the available models. Pick one, fill in what you know about the car, and submit. Year, brand, fuel type, seller type, transmission, ownership, and seat count are required; distance driven, mileage, engine size, and maximum power can be left blank and are filled in with the median value from the training data.

Inputs are bounded by what is physically possible rather than by the ranges present in the training data, so an unusual but real car is not refused for being unusual. Year is the one exception: the form accepts 2000 onward, because the model is not reliable below that. See Limitations.

## Running it

```
git clone https://github.com/khantmin-km/car-price-prediction.git
cd car-price-prediction
uv sync
uv run python app/code/main.py
```

Then open <http://localhost:8050>.

In a container instead:

```
docker compose -f app/docker-compose.yaml up -d
```

Settings come from environment variables, listed with their defaults in `.env.example`. No `.env` file is needed to run locally.

## Project layout

```
app/
	.Dockerfile                        multi-stage build on python:3.11-slim
	docker-compose.yaml                local run
	code/
		main.py                        entry point and navigation bar
		config.py                      environment-variable settings
		options.py                     dropdown options and input bounds
		pages/                         one module per model page
		models/                        trained model bundles (.joblib)
	src/
		preprocessing.py               cleaning and feature encoding
	notebooks/                         analysis, model selection, and the written report
	data/raw/cars.csv                  the dataset
```

`src/preprocessing.py` is imported by both the notebooks and the application, so the cleaning and encoding used to train a model are the same ones used to serve it. A trained model is saved as a bundle holding the fitted estimator, the imputation medians, the expected feature columns, and the category lists the form offers.

## Models

| Model | Held-out test R² |
| --- | --- |
| Linear Regression (deployed) | 0.89 |

Random Forest scored higher in cross-validation (0.91 against 0.86) but lower on the held-out test set (0.84), because it cannot predict a price above anything it saw during training. The notebook sets out the comparison in full.

## Notebooks

`notebooks/A1_car_price_prediction.ipynb` holds the cleaning, exploratory analysis, model selection, test-set evaluation, feature importance, and the written report.

## Limitations

Accuracy falls as cars get older and breaks down before 2000, which is why the form refuses those years. Missing values are filled with a global median, so estimates for sparsely-filled listings are biased upward. Brands with only a few listings are fitted unreliably.  These are set out with supporting figures in the "Limitations and design decisions" section of the notebook.