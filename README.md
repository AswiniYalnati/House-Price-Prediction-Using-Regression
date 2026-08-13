# USA House Price Prediction

A machine learning project that predicts house prices using the **USA Housing dataset** and compares the performance of multiple regression algorithms. The project covers data preprocessing, model training, evaluation, model serialization, and deployment through a **Flask web application**.

## Project Overview

The objective of this project is to build a reliable house price prediction system by training and comparing different machine learning regression models. The dataset contains features such as average area income, house age, number of rooms, number of bedrooms, and area population.

The project implements and evaluates **13 regression models**:

* Linear Regression
* Robust Regression
* Ridge Regression
* Lasso Regression
* Elastic Net
* Polynomial Regression
* SGD Regressor
* Artificial Neural Network (ANN)
* Random Forest Regressor
* Support Vector Regression (SVM)
* LightGBM
* XGBoost
* K-Nearest Neighbors (KNN)

Each model is trained on the training dataset and evaluated on the test dataset using **Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score**.

## Key Features

* Data preprocessing and feature selection
* Train-test splitting with a fixed random state
* Training of multiple regression algorithms
* Model performance comparison using MAE, MSE, and R²
* Saving trained models as `.pkl` files using Pickle
* Saving model evaluation results to a CSV file
* Flask-based web application for house price prediction
* Ability to select a trained model and generate predictions
* Separate page for viewing model evaluation results

The Flask application loads the trained models and allows users to enter housing features and select a regression model to obtain a predicted house price.

## Technologies Used

* **Python**
* **Pandas**
* **Scikit-learn**
* **LightGBM**
* **XGBoost**
* **Flask**
* **Pickle**
* **HTML/CSS**
* **Machine Learning – Regression**

## Evaluation Metrics

The models are evaluated using:

* **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted prices.
* **MSE (Mean Squared Error):** Measures the average squared prediction error.
* **R² Score:** Measures how well the model explains the variation in house prices.

The evaluation results are stored in `model_evaluation_results.csv` for easy comparison.

## Project Structure

```text
USA-House-Price-Prediction/
│
├── USA_Housing.csv
├── House_regressor.py
├── model.py
├── app.py
│
├── LinearRegression.pkl
├── RobustRegression.pkl
├── RidgeRegression.pkl
├── LassoRegression.pkl
├── ElasticNet.pkl
├── PolynomialRegression.pkl
├── SGDRegressor.pkl
├── ANN.pkl
├── RandomForest.pkl
├── SVM.pkl
├── LGBM.pkl
├── XGBoost.pkl
├── KNN.pkl
│
├── model_evaluation_results.csv
│
├── templates/
│   ├── index.html
│   ├── results.html
│   └── model.html
│
└── README.md
```

## How It Works

1. Load the USA Housing dataset.
2. Separate input features and the target variable (`Price`).
3. Split the dataset into training and testing sets.
4. Train multiple regression models.
5. Generate predictions using the test data.
6. Calculate MAE, MSE, and R² scores.
7. Save the trained models as Pickle files.
8. Save the evaluation results as a CSV file.
9. Load the trained models into the Flask application.
10. Allow users to enter house details and select a model to predict the house price.

The trained models are serialized individually as Pickle files, while the evaluation results are saved as a CSV file.

## Future Improvements

* Add data visualization and exploratory data analysis.
* Perform hyperparameter tuning for better model performance.
* Add feature scaling and advanced feature engineering.
* Improve the Flask UI with a responsive design.
* Deploy the application using a cloud platform.
* Add interactive charts for comparing model performance.

## Conclusion

This project demonstrates how different machine learning regression techniques can be applied to **house price prediction** and compared using standard evaluation metrics. It also demonstrates how trained machine learning models can be integrated into a **Flask web application** to provide real-time predictions.
