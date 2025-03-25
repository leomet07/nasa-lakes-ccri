# Setup dotenv
from dotenv import load_dotenv
import os

load_dotenv()

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from scipy import stats
import joblib
import math
from matplotlib.ticker import ScalarFormatter

GRAPH_AND_COMPARE_PERFORMANCE = os.getenv("GRAPH_AND_COMPARE_PERFORMANCE").lower() == "true"
PERFORMANCE_CHART_PATH = os.getenv("PERFORMANCE_CHART_PATH") or "charts"
USE_CACHED_MODEL = os.getenv("USE_CACHED_MODEL").lower() == "true"
CPU_MODEL_SAVE_FILE = "model_cpu.joblib"

import model_data
training_data = model_data.training_data
# cleaned_data = cleaned_data.replace(-99999.0, np.nan) # Not needed, -99999 working fine

X = training_data.drop(columns=['chl_a'])
y = training_data['chl_a']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=model_data.RANDOM_STATE)
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
mae = mean_absolute_error(y_test, y_pred)

print(f"r2 score: {r2}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")

if GRAPH_AND_COMPARE_PERFORMANCE:
    plt.rcParams.update({'font.size': 18})

    if not os.path.exists(PERFORMANCE_CHART_PATH):
        os.makedirs(PERFORMANCE_CHART_PATH)
    print("mean of test: ", np.mean(y_test))
    print("mean of pred: ", np.mean(y_pred))
    print("min of test: ", np.min(y_test))
    print("min of pred: ", np.min(y_pred))


    plt.figure("Number of Chl-a Samples Per Lake", (7,7))

    insitu_lakeids = np.unique(model_data.all_data_cleaned["lagoslakei"])
    num_samples_in_each_lake = [len(model_data.all_data_cleaned[model_data.all_data_cleaned["lagoslakei"] == lakeid]) for lakeid in insitu_lakeids]
    
    plt.hist(num_samples_in_each_lake, 75, color="skyblue", ec="black")
    plt.xlim(0, 200) # 0 to 200 samples per lake
    plt.xticks(np.arange(0, 200, 20))
    # plt.ylabel("Frequency", fontweight='bold') # label not needed becauase in the paper it is next to a chart that alr has this ylabel
    plt.xlabel("Number of Samples per Lake", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PERFORMANCE_CHART_PATH, "number_of_samples_per_lake.png"), bbox_inches='tight')
    

    # Plot y_pred performance: https://stackoverflow.com/questions/19064772/visualization-of-scatter-plots-with-overlapping-points-in-matplotlib
    plt.figure("Comparing Model's Prediction on Testing In-Situ Dataset to Corresponding Measured Value", (14,7))
    values = np.vstack([y_test, y_pred])
    kernel = stats.gaussian_kde(values, bw_method=.02)(values)

    plt.scatter(y_test, y_pred, s=20, c=kernel,cmap='viridis')
    plt.axline((0,0), (50,50), linewidth=2, color='red')
    colorbar = plt.colorbar()
    colorbar.set_label("Density", rotation=270, labelpad=15, fontweight='bold')

    plt.xscale('log')
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.ticklabel_format(axis='x', style='plain') # so that xticks are written in decimal
    plt.yscale('log')
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.ticklabel_format(axis='y', style='plain') # so that yticks are written in decimal

    plt.xlim(0.1, 100) # starts at 0.1 bc this is LOG scale and 0 is invalid
    plt.ylim(0.1, 100) # starts at 0.1 bc this is LOG scale and 0 is invalid
    plt.xlabel('Observed Chl-a (µg/L)', fontweight='bold')
    plt.ylabel('Predicted Chl-a (µg/L)', fontweight='bold')
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(PERFORMANCE_CHART_PATH, "scatter_plot_pred_vs_real.png"), bbox_inches='tight')

    # Feature importances plot
    plt.figure('Feature Importances', (14,7))
    feature_importances = andrew_model.feature_importances_
    indices = np.argsort(feature_importances)
    features = X.columns
    features = features.str.replace('pct_dev', f'Developed')
    features = features.str.replace('pct_ag', f'Agriculture')
    features = features.str.replace('SA_SQ_KM', f'Surface area')
    # plt.barh(range(len(indices)), feature_importances[indices], color='b', align='center') # Sorted
    # plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.barh(features, feature_importances, color="skyblue", ec="black")
    plt.xlim(0, 0.3)
    plt.xlabel('Relative Importance', fontweight='bold')
    # plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.tight_layout()
    plt.savefig(os.path.join(PERFORMANCE_CHART_PATH, "feature_importances.png"))

    plt.figure(f"Chlorophyll-a Prediction on Testing Portion of In-Situ Data", (14,7))
    # Histogram of model's performance on testing data, aka 20% of the in situ data
    plt.hist(y_pred, 100, color="skyblue", ec="black")
    plt.ylabel("Frequency", fontweight='bold')
    plt.xlabel("Predicted Chl-a (µg/L)", fontweight='bold')
    plt.xticks(np.arange(0, 60, 5.0))
    plt.xlim((0, 60))
    plt.tight_layout()
    plt.savefig(os.path.join(PERFORMANCE_CHART_PATH, "histogram_chla_prediction_testing_part_of_insitu.png"), bbox_inches='tight')


    plt.figure(f"Chlorophyll-a Prediction Error on Testing Portion of In-Situ Data", (14,7))
    # Histogram of model's absolute error on testing data, aka 20% of the in situ data
    y_error = y_pred - y_test
    plt.hist(y_error, 100, color="skyblue", ec="black")
    plt.ylabel("Frequency", fontweight='bold')
    plt.xlabel("Predicted Chl-a Error (µg/L)", fontweight='bold')
    plt.xticks(np.arange(-60, 60, 5.0))
    plt.xlim((-60, 60))
    plt.tight_layout()
    plt.savefig(os.path.join(PERFORMANCE_CHART_PATH, "histogram_chla_prediction_error_testing_part_of_insitu.png"), bbox_inches='tight')


    # Run predictions on ALL insitu data
    all_insitu_predicted = andrew_model.predict(X.values)

    plt.figure(f"Chlorophyll-a Prediction on All In-Situ Data", (14,7))
    # Histogram of model's performance on 100% of the in situ data
    plt.hist(all_insitu_predicted, 100, color="skyblue", ec="black")
    plt.ylabel("Frequency", fontweight='bold')
    plt.xlabel("Predicted Chl-a (µg/L)", fontweight='bold')
    plt.xticks(np.arange(0, 60, 5.0))
    plt.xlim((0, 60))
    plt.tight_layout()
    plt.savefig(os.path.join(PERFORMANCE_CHART_PATH, "histogram_chla_prediction_all_insitu.png"), bbox_inches='tight')

    
    plt.figure(f"Chlorophyll-a Prediction Error on All In-Situ Data", (14,7))
    # Histogram of model's absolute error on 100% of the in situ data
    all_insitu_predicted_error = all_insitu_predicted - y.values
    plt.hist(all_insitu_predicted_error, 100, color="skyblue", ec="black")
    plt.ylabel("Frequency", fontweight='bold')
    plt.xlabel("Predicted Chl-a Error (µg/L)", fontweight='bold')
    plt.xticks(np.arange(-60, 60, 5.0))
    plt.xlim((-60, 60))
    plt.tight_layout()
    plt.savefig(os.path.join(PERFORMANCE_CHART_PATH, "histogram_chla_prediction_error_all_insitu.png"), bbox_inches='tight')

    plt.figure(f"Histogram of In-Situ Chlorophyll-a", (7,7))
    insitu_chla_median = np.median(y.values)
    # Histogram of the in situ data chla
    plt.hist(y.values, 100, color="skyblue", ec="black")
    plt.ylabel("Frequency", fontweight='bold')
    plt.xlabel("In Situ Chl-a (µg/L)", fontweight='bold')
    plt.xticks(np.arange(0, 60, 5.0))
    plt.xlim((0, 60))
    plt.axvline(x=insitu_chla_median, color='black', linestyle='--', linewidth=1.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PERFORMANCE_CHART_PATH, "histogram_insitu_chla.png"), bbox_inches='tight')

    plt.figure(f"Predicted vs Insitu (Testing Dataset)", (14,7))
    # Overlayed Histogram of the in situ truths vs the predicted values
    plt.hist(y_test, 200, label="Insitu Chl-a Frequency", histtype="stepfilled", alpha=0.6)
    plt.hist(y_pred, 200, label="Predicted Chl-a Frequency", histtype="stepfilled", alpha=0.6)
    plt.legend()
    plt.ylabel("Frequency", fontweight='bold')
    plt.xlabel("In Situ Chl-a (µg/L)", fontweight='bold')
    plt.xticks(np.arange(0, 60, 5.0))
    plt.xlim((0, 60))
    plt.tight_layout()
    plt.savefig(os.path.join(PERFORMANCE_CHART_PATH, "overlay_predicted_and_insitu_testing_part_of_insitu_full.png"), bbox_inches='tight')
   
    # Same thing as above, but split x_axis
    fig, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w', figsize=(14,7))
    fig.supylabel("Frequency", fontweight='bold')
    fig.supxlabel("In Situ Chl-a (µg/L)", fontweight='bold')
    fig.canvas.manager.set_window_title("Split X-axis, Predicted vs Insitu (Testing Dataset)")
    plt.xticks(np.arange(0, 70, 5))
    plt.tight_layout()

    ax.hist(y_test, 200, label="Insitu Chl-a Frequency", histtype="stepfilled", alpha=0.6)
    ax.hist(y_pred, 200, label="Predicted Chl-a Frequency", histtype="stepfilled", alpha=0.6)
    ax2.hist(y_test, 200, label="Insitu Chl-a Frequency", histtype="stepfilled", alpha=0.6)
    ax2.hist(y_pred, 200, label="Predicted Chl-a Frequency", histtype="stepfilled", alpha=0.6)

    ax2.legend()
    ax.set_xlim(0, 25)
    ax2.set_xlim(55, 70)
    # hide the spines between ax and ax2
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.yaxis.tick_left()
    # ax.tick_params(labelright='off')
    ax2.yaxis.tick_right()

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
    ax2.plot((-d, +d), (-d, +d), **kwargs)


    plt.savefig(os.path.join(PERFORMANCE_CHART_PATH, "overlay_predicted_and_insitu_testing_part_of_insitu.png"), bbox_inches='tight')

    # Show both at the same time
    plt.show()