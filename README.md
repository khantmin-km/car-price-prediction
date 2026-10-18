# Car Price Prediction

A web application that estimates the selling price of a used car from its year, distance driven, brand, fuel type, and a handful of other details. It offers three models trained on 6,827 cleaned listings: a scikit-learn linear regression, a polynomial regression trained with our own gradient-descent class, and our own logistic regression that puts a car into one of four price ranges.

## What it does

The navigation bar lists the available models: **Linear Model**, **Polynomial Model**, and **Price Class**. Pick one, fill in what you know about the car, and submit. Year, brand, fuel type, seller type, transmission, ownership, and seat count are required; distance driven, mileage, engine size, and maximum power can be left blank and are filled in with the median value from the training data.

Inputs are bounded by what is physically possible rather than by the ranges present in the training data, so an unusual but real car is not refused for being unusual. Year is the one exception: the form accepts 2000 onward, because the model is not reliable below that. The Polynomial Model also shows a warning above about 280 bhp, because almost no car in the training data is that powerful. See Limitations.

## Running it

```
git clone https://github.com/khantmin-km/car-price-prediction.git
cd car-price-prediction
uv sync
uv run python app/code/main.py
```

Then open [http://localhost:8050](http://localhost:8050/).

In a container instead:

```
docker compose -f app/docker-compose.yaml up -d
```

Settings come from environment variables, listed with their defaults in `.env.example`. No `.env` file is needed to run locally.

## Project layout

```
app/
├── .Dockerfile                   # Multi-stage build on python:3.11-slim
├── docker-compose.yaml           # Local run
├── docker-compose.deploy.yaml    # Server run, behind Traefik
├── code/
│   ├── main.py                   # Entry point and navigation bar
│   ├── config.py                 # Environment-variable settings
│   ├── options.py                # Dropdown options and input bounds
│   ├── car_form.py               # Input form shared by all model pages
│   ├── pages/                    # One module per model page
│   └── models/                   # Trained model bundles
├── src/
│   ├── preprocessing.py          # Cleaning, feature encoding, scaling
│   ├── linear_regression.py      # Gradient-descent regression class
│   ├── logistic_regression.py    # Gradient-descent classification class with its own metrics
│   ├── regularization.py         # Lasso and Ridge penalties
│   └── experiment.py             # Cross-validation and MLflow logging
├── notebooks/                    # Analysis, model selection, experiments
├── tests/
│   └── test_model.py             # Unit tests for the Price Class model
├── .github/workflows/
│   └── ci-cd.yml                 # Test, build and push the image, deploy
└── data/
    └── raw/
        └── cars.csv              # Dataset
```

`src/preprocessing.py` is imported by both the notebooks and the application, so the cleaning and encoding used to train a model are the same ones used to serve it. A trained model is saved as a bundle holding the fitted estimator, the imputation medians, the expected feature columns, and the category lists the form offers. The Polynomial Model's bundle also holds the scaling values learned from the training data.

## Models

| Model                             | Page             | Cross-validation R² (same 5 folds) | Held-out test R² |
| --------------------------------- | ---------------- | ---------------------------------- | ---------------- |
| Linear Regression (scikit-learn)  | Linear Model     | 0.86                               | 0.89             |
| Polynomial Regression (own class) | Polynomial Model | 0.89                               | 0.83             |

**Linear Model.** Random Forest scored higher in cross-validation (0.91 against 0.86) but lower on the held-out test set (0.84), because it cannot predict a price above anything it saw during training. Linear Regression was chosen instead.

**Polynomial Model.** Our own `LinearRegression` class, trained by gradient descent. It supports batch, mini-batch, and stochastic updates, zero or Xavier starting weights, and momentum, plus Lasso and Ridge penalties. We compared four model types (normal, Lasso, Ridge, polynomial) across every combination of update method, starting weights, learning rate (0.01, 0.001, 0.0001), and momentum on or off: 144 settings, each scored with 5-fold cross-validation and logged in MLflow. The best was degree-2 polynomial regression with batch updates, Xavier starting weights, learning rate 0.01, and momentum 0.9.

On the same folds, the Polynomial Model beats the Linear Model in all five, and its typical error is 14.6% of the price against 15.7%. It scores lower on the test set because of a single car: a 400 bhp Volvo, far more powerful than anything in the training data. The curve flattens out at high power, so the model guesses less than half its price. Without that car the two models score the same (0.886 against 0.887). The notebooks set out both comparisons in full.

**Price Class.** Our own `LogisticRegression` class, trained by gradient descent with batch, mini-batch, or stochastic updates and an optional Ridge penalty. Accuracy, precision, recall, F1, and their macro and weighted averages are methods of the class, and its report matches scikit-learn's `classification_report`. Prices are split into four classes at the quartiles of the training prices: below ₹250,000, ₹250,000 to under ₹415,000, ₹415,000 to under ₹650,000, and ₹650,000 and up. Of eight settings compared on a validation slice of the training data, the best was mini-batch updates with learning rate 0.5 and no Ridge penalty. On the test set it scores 0.71 accuracy and 0.71 macro F1, and 388 of its 396 mistakes are only one class off.

## Notebooks

- `notebooks/A1_car_price_prediction.ipynb` holds the cleaning, exploratory analysis, model selection, test-set evaluation, feature importance, and the written report for the Linear Model.
- `notebooks/A2_car_price_prediction.ipynb` reuses that notebook's data preparation and replaces the modelling with our own class. It holds the check against scikit-learn, the 144-setting experiment, the results, the test-set evaluation, feature importance, and the written report for the Polynomial Model.
- `notebooks/A3_car_price_classification.ipynb` reuses the same data preparation, turns the price into four classes, and trains our own logistic regression. It holds the metric check against scikit-learn, the eight runs with and without Ridge, the test-set evaluation, and the written report for the Price Class model.

## Experiment tracking

The experiments are logged to a local MLflow store: `mlflow.db` and `mlartifacts/` at the repository root, in experiments `st127634-a2` and `st127634-a3`. The first run of the A2 notebook rebuilds its experiment, which takes about an hour. After that, it reuses the logged runs and finishes in under a minute. The A3 notebook runs in under a minute. Its chosen model is registered as `st127634-a3-model` with the alias `staging`. It is logged locally and the notebook shows screenshots of the runs and the registered model. To browse the runs:

```
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

## Tests and CI/CD

```
uv run --group test pytest
```

`tests/test_model.py` loads the saved Price Class model and checks that (1) it takes the expected input, prepared the same way the web app prepares it, and (2) it returns one class from 0 to 3 for each car.

Every push runs `.github/workflows/ci-cd.yml` on GitHub Actions. It runs the tests first. If they pass, it builds the image and pushes it to Docker Hub as `khantminn/car-price-prediction:a3`. If that succeeds, it connects to the course server over SSH and restarts the app with the new image.

## Limitations

Accuracy falls as cars get older and breaks down before 2000, which is why the form refuses those years. Missing values are filled with a global median, so estimates for sparsely-filled listings are biased upward. Brands with only a few listings are fitted unreliably. The Polynomial Model should not be trusted for very powerful cars, because the training data has almost none above 280 bhp. The Price Class model is weakest on the two middle classes (F1 0.64 and 0.58, against 0.83 and 0.80 for the cheapest and most expensive), where neighbouring prices are easy to confuse. These are set out with supporting figures in the "Limitations and design decisions" section of each notebook.
