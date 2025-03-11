# Setup dotenv
from dotenv import load_dotenv
import os

load_dotenv()

from scipy import stats
import joblib
import cudf
from cuml.ensemble import RandomForestRegressor
from dask_ml.model_selection import RandomizedSearchCV
from cuml.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import time
import numpy as np
from matplotlib import pyplot as plt

DO_HYPERPARAM_SEARCH = os.getenv("DO_HYPERPARAM_SEARCH").lower() == "true"
GRAPH_AND_COMPARE_PERFORMANCE = os.getenv("GRAPH_AND_COMPARE_PERFORMANCE").lower() == "true"
USE_CACHED_MODEL = os.getenv("USE_CACHED_MODEL").lower() == "true"
GPU_MODEL_SAVE_FILE = "model_gpu.joblib"

import model_data

training_data = cudf.from_pandas(model_data.training_data)

for col in training_data.select_dtypes(["object"]).columns:
    training_data[col] = training_data[col].astype("category").cat.codes.astype(np.int32)

# cast all columns to int32
for col in training_data.columns:
    training_data[col] = training_data[col].astype(np.float32)  # float32 type needed for cuml random forest

X = training_data.drop(columns=['chl_a'])
y = training_data['chl_a']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=model_data.RANDOM_STATE)
print("Dataframes created and data split successfully.")

def hyper_param_search_and_train_model():
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features =  ['log2', 'sqrt', 1.0] # all in the following array are fast but log2 is fastest ['log2', 'sqrt', 1.0]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    # max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [4,7,10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [4,7,10]

    # Define the parameter grid for RandomizedSearchCV
    param_grid = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'split_criterion': [2] # 2 is MSE (source: https://stackoverflow.com/questions/77984286/cuml-randomforestclassifier-typeerror-an-integer-is-required)
    }


    # Initialize the Random Forest model
    rf_model = RandomForestRegressor() #2 for mse

    time_start = time.time()
    print("Searching for best hyperparamaters using a random search and checking by fitting model...")

    # Initialize the RandomizedSearchCV with 5-fold cross-validation
    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_grid,
        n_iter=5,
        scoring='neg_mean_absolute_error', # neg_mean_absolute_error = MAE
        cv=10, # cv of 10 has the same accuracy but runs slower
        #verbose=2,
        # random_state=42,
        n_jobs=-1
    )

    # Fit the RandomizedSearchCV to find the best hyperparameters (by randomly trying combos within param grid and fitting and testing accordingly)
    random_search.fit(X_train.to_numpy(), y_train.to_numpy())
    time_end = time.time()
    time_diff = time_end - time_start
    print(f"Random search & Fit finished, elapsed {time_diff} seconds")

    # Get the best parameters
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_
    return best_params, best_model
   
def train_gpu_model():
    if DO_HYPERPARAM_SEARCH:
        best_params, model = hyper_param_search_and_train_model()
        print(f"Best Parameters: {best_params}")
    else:
        preset_params = {
            'max_depth': 30, # Andrew params
            'max_features': 'sqrt',
            'min_samples_leaf' :1,
            'min_samples_split' :2,
            'n_estimators':1200,
        }

        print("Known fit starting...")
        model = RandomForestRegressor(**preset_params) # Instead of searching for params, use preconfigured params andrew found
        time_start = time.time()
        model.fit(X_train.to_numpy(), y_train.to_numpy())  # Fit model (which uses preconfigured params)
        time_end = time.time()
        time_diff = time_end - time_start
        print(f"Known fit finished, elapsed {time_diff} seconds")

    joblib.dump(model, GPU_MODEL_SAVE_FILE)
    return model

if USE_CACHED_MODEL and os.path.exists(GPU_MODEL_SAVE_FILE):
    print("Loading (GPU) model from cache...")
    andrew_model = joblib.load(GPU_MODEL_SAVE_FILE)
    print("Loaded!")
else:
    print("Training (GPU) model from scratch...")
    andrew_model = train_gpu_model()
    print("Trained!")

time_start = time.time()
print("Predicting on test...")
# Predict on the test set using the best model
y_pred = andrew_model.predict(X_test)
time_end = time.time()
time_diff = time_end - time_start
print(f"Predicted! Elapsed {time_diff} seconds")

# Calculate the Mean Squared Error and r2
r2 = r2_score(y_test.to_numpy(), y_pred.to_numpy())
rmse = mean_squared_error(y_test.to_numpy(), y_pred.to_numpy()) ** 0.5
mae = mean_absolute_error(y_test.to_numpy(), y_pred.to_numpy())

print(f"r2 score: {r2}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")


if GRAPH_AND_COMPARE_PERFORMANCE:
    print("mean of test: ", np.mean(y_test))
    print("mean of pred: ", np.mean(y_pred))

    plt.figure(1) # Figure 1
    plt.hist(y_pred, 100)
    plt.xlim(0, 75)
    plt.xlabel("Predicted chl-a (ug/l)")
    plt.ylabel("Frequency")
    plt.title("Prediction Histogram (GPU)")

    # Plot y_pred performance: https://stackoverflow.com/questions/19064772/visualization-of-scatter-plots-with-overlapping-points-in-matplotlib
    values = np.vstack([y_test.to_numpy(), y_pred.to_numpy()])
    kernel = stats.gaussian_kde(values, bw_method=.02)(values)
    plt.figure(2) # Figure 1
    plt.scatter(y_test.to_numpy(), y_pred.to_numpy(), s=20, c=kernel,cmap='viridis')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Observed Chl-a (ug/l)')
    plt.ylabel('Predicted Chl-a (ug/l)')
    plt.title('GPU Random Forest Regression')

    # Show both figures at once
    plt.show()