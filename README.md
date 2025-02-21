# Emergency Response Time Prediction

This repository contains code for predicting emergency response times using machine learning. It leverages data related to emergency events, vehicle locations, and routing information from OSRM (Open Source Routing Machine).

## Overview

The goal is to build a model that accurately predicts either:

*   `delta selection-departure`: The time between emergency vehicle selection and departure.
*   `delta departure-presentation`: The time between emergency vehicle departure and arrival at the event location.

The code is structured into modules for data loading, feature engineering, model selection, training, and evaluation, promoting maintainability and scalability.

## File Structure
```bash
emergency_response_prediction/
├── data/
│ ├── x_train.csv
│ ├── x_test.csv
│ └── y_train_u9upqBE.csv
├── src/
│ ├── data_loader.py # Loads data from CSV files.
│ ├── feature_engineering.py # Creates and preprocesses features.
│ ├── model_selection.py # Splits data into training and validation sets.
│ ├── model_training.py # Trains and evaluates machine learning models.
│ ├── utils.py # Utility functions (e.g., logging).
├── config.py # Configuration file for paths and parameters.
├── requirements.txt # List of Python dependencies.
├── README.md # This file.
```
## Dependencies

The following Python libraries are required:

*   pandas
*   numpy
*   scikit-learn
*   geopy
*   torch

Install the dependencies using pip:

```
pip install -r requirements.txt
```

The following data files are expected in the data/ directory:

`x_train.csv`: Training data features.

`x_test.csv`: Testing data features.

`y_train_u9upqBE.csv`: Training data target variables.

## Usage

* Clone the repository:
```
git clone [repository_url]
cd emergency_response_prediction
```

* Install dependencies:

```
pip install -r requirements.txt
```

Place the data files (`x_train.csv`, `x_test.csv`, `y_train_u9upqBE.csv`) in the `data/` directory.
**Note: for copy right purposes the data has been removed form this repository**

Run the training script:

```
python src/model_training.py
```

This script will:

* Load and preprocess the data.

* Split the training data into training and validation sets.

* Train several machine learning models (Linear Regression, Random Forest, PyTorch Neural Network).

* Evaluate the models on the validation set.

* Select the best model based on Mean Absolute Error (MAE).

* Evaluate the best model on the test data.

* Print evaluation metrics (MAE, RMSE, R-squared).

## Configuration

The config.py file contains configuration parameters such as:

* Data file paths.

* Feature names (categorical, numerical, temporal).

* Target variable.

* Random seed for reproducibility.

* Test set size for validation split.

Modify this file to customize the project to your specific data and requirements.

## Feature Engineering

* The `feature_engineering.py` module performs feature engineering steps, including:

* Calculating the Haversine distance between GPS coordinates.

* Creating temporal features from datetime columns (hour, day of week, month).

* Handling missing values using imputation.

* Encoding categorical features using one-hot encoding.

* Scaling numerical features using standardization.

Customize this module to add or modify features as needed. Consider adding features based on OSRM data.

## Model Training

The `model_training.py` module trains and evaluates machine learning models. It currently includes:

* Linear Regression

* Random Forest Regressor

* A simple PyTorch Neural Network

You can add more models or modify the existing ones. The module also includes functions for model evaluation and selection based on validation performance.

## Evaluation Metrics

The following evaluation metrics are used:

* Mean Absolute Error (MAE)

* Root Mean Squared Error (RMSE)

* R-squared

## Next Steps

* **Customize Feature Engineering:** Adapt the feature_engineering.py module to create more relevant features for your data. Focus on leveraging OSRM data and creating interaction features.

* **Hyperparameter Tuning:** Use techniques like GridSearchCV or RandomizedSearchCV from scikit-learn to optimize model hyperparameters.

* **Explore Advanced Models:** Experiment with more sophisticated models, such as Gradient Boosting Machines (XGBoost, LightGBM, CatBoost) or more complex Deep Learning architectures.

* **Experiment Tracking:** Use tools like MLflow or TensorBoard to track your experiments and compare different models.

* **Deployment:** Consider how you would deploy the model to a production environment.

* **Address Class Imbalance (if present):** Explore techniques like SMOTE or class weighting.

* **Survival Analysis:** If you want to consider the 'delta departure-presentation' as a event data, use it to Survival Analysis and models.

* **Geospatial Visualizations:** Use libraries like geopandas and folium to visualize emergency events and vehicle routes on a map.

## Contributing

Contributions are welcome! Please submit pull requests with bug fixes, new features, or improved documentation.

