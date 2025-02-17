# Setup dotenv
from dotenv import load_dotenv
import os

load_dotenv()

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from scipy import stats
import joblib


GRAPH_AND_COMPARE_PERFORMANCE = os.getenv("GRAPH_AND_COMPARE_PERFORMANCE").lower() == "true"
USE_CACHED_MODEL = os.getenv("USE_CACHED_MODEL").lower() == "true"
CPU_MODEL_SAVE_FILE = "model_cpu.joblib"

import model_data
cleaned_data = model_data.cleaned_data
# cleaned_data = cleaned_data.replace(-99999.0, np.nan) # Not needed, -99999 working fine

X = cleaned_data.drop(columns=['chl_a'])
y = cleaned_data['chl_a']

print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=621)
print("Dataframes created and data split successfully.")

def train_cpu_model():
    andrew_params = {
        'max_depth': 30, # Andrew params
        'max_features': 'sqrt',
        'min_samples_leaf' :1,
        'min_samples_split' :2,
        'n_estimators':1200,
    }

    print("Known fit starting...")
    model = RandomForestRegressor(**andrew_params) # Instead of searching for params, use preconfigured params andrew found
    time_start = time.time()
    model.fit(X_train.values, y_train)  # Fit model (which uses preconfigured params)
    time_end = time.time()
    time_diff = time_end - time_start
    print(f"Known fit finished, elapsed {time_diff} seconds")

    joblib.dump(model, CPU_MODEL_SAVE_FILE)
    return model

if USE_CACHED_MODEL and os.path.exists(CPU_MODEL_SAVE_FILE):
    print("Loading (CPU) model from cache...")
    andrew_model = joblib.load(CPU_MODEL_SAVE_FILE)
    print("Loaded!")
else:
    print("Training (CPU) model from scratch...")
    andrew_model = train_cpu_model()
    print("Trained!")

time_start = time.time()
print("Predicting on test...")
# Predict on the test set using the best model
y_pred = andrew_model.predict(X_test.values)
time_end = time.time()
time_diff = time_end - time_start
print(f"Predicted! Elapsed {time_diff} seconds")

# Calculate the Mean Squared Error and r2
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print(f"r2 score: {r2}")
print(f"RMSE: {rmse}")


if GRAPH_AND_COMPARE_PERFORMANCE:
    print("mean of test: ", np.mean(y_test))
    print("mean of pred: ", np.mean(y_pred))
    print("min of test: ", np.min(y_test))
    print("min of pred: ", np.min(y_pred))
    # Model analysis
    plt.figure(1)
    plt.hist(y_pred, 200)
    plt.xlabel("Predicted chl-a (ug/l)")
    plt.ylabel("Frequency")
    plt.title("Prediction Histogram (CPU)")


    # Plot y_pred performance: https://stackoverflow.com/questions/19064772/visualization-of-scatter-plots-with-overlapping-points-in-matplotlib
    plt.figure(2)
    values = np.vstack([y_test, y_pred])
    kernel = stats.gaussian_kde(values, bw_method=.02)(values)

    plt.scatter(y_test, y_pred, s=20, c=kernel,cmap='viridis')
    # plt.axline((0,0), (50,50), linewidth=1, color='black')
    # plt.axis((0,50,0,50))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Observed Chl-a (ug/l)')
    plt.ylabel('Predicted Chl-a (ug/l)')
    plt.title('CPU Random Forest Regression')


    # Feature importances plot
    plt.figure(3)
    feature_importances = andrew_model.feature_importances_
    indices = np.argsort(feature_importances)
    features = X.columns
    plt.title('Feature Importances')
    # plt.barh(range(len(indices)), feature_importances[indices], color='b', align='center') # Sorted
    # plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.barh(features, feature_importances)
    # plt.yticks(range(len(indices)), [features[i] for i in indices])

    plt.xlabel('Relative Importance')
    plt.show()

    # Show both at the same time
    plt.show()