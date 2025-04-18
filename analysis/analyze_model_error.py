import db_utils
import raster_utils
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from datetime import datetime

load_dotenv()
ROOT_DB_FILEPATH = os.getenv("ROOT_DB_FILEPATH")  # for accessing files manually
USE_TEST_DATASET_FOR_ERROR_ANALYSIS = (
    os.getenv("USE_TEST_DATASET_FOR_ERROR_ANALYSIS") == "true"
)
TIF_OUT_FILEPATH = os.getenv("TIF_OUT_FILEPATH")  # for accessing files manually

all_spatial_predictions = list(db_utils.spatial_predictions_collection.find({}))
all_spatial_predictions_df = pd.DataFrame.from_dict(all_spatial_predictions)
all_spatial_predictions_df["date_str"] = pd.to_datetime(
    all_spatial_predictions_df["date"], format=r"%Y-%m-%d"
)

print(all_spatial_predictions_df.head)

all_data = pd.read_csv(
    os.getenv("ALL_INPUT_DATA_CSV")
)  # this later creates cleaned_data for training model

all_data_train, all_data_test = train_test_split(
    all_data, test_size=0.2, random_state=621
)  # constant hard coded from ../ml_model/model_data.py

if USE_TEST_DATASET_FOR_ERROR_ANALYSIS:
    all_data = all_data_test

abs_errors = []
squared_errors = []

lakeids = []

for index, row in tqdm(all_data.iterrows(), total=len(all_data)):
    lagoslakeid = row["lagoslakei"]
    lat = row["MEAN_lat"]
    lng = row["MEAN_long"]
    true_chl_a = row["chl_a"]

    date = row["date"]  # satelltie fly over date ("sample_date" is for sample_date)

    matched_predictions = all_spatial_predictions_df[
        (all_spatial_predictions_df["lagoslakeid"] == lagoslakeid)
        & (all_spatial_predictions_df["date_str"] == date)
    ]  # filter to find lake's spatial prediction with correct date

    if len(matched_predictions) == 0:
        continue  # did not find any prediction image with insitu date

    # print(f"Found prediction for Lake{lagoslakeid} on {date}")

    spatial_prediction = matched_predictions.iloc[0]
    file_path = os.path.join(TIF_OUT_FILEPATH, spatial_prediction["raster_image"])
    try:
        stats = raster_utils.get_analytics_from_circular_section_in_raster_file(
            file_path, lat, lng
        )

        prediction_mean_val = stats[2]
        error = prediction_mean_val - true_chl_a

        abs_error = abs(error)
        squared_error = abs_error**2

        abs_errors.append(abs_error)
        squared_errors.append(squared_error)

        lakeids.append(lagoslakeid)

        # print(
        #     "True Value: ",
        #     true_chl_a,
        #     "Prediction (mean) value: ",
        #     prediction_mean_val,
        #     "Error: ",
        #     error,
        # )
    except ValueError as e:
        # ValueError: zero-size array to reduction operation fmax which has no identity
        # Means only nans or -infs in circle raster which got filtered out
        if str(e).startswith("zero-size array"):
            pass
            # print(f"Only NAN values within circle about point. Was there a cloud?")
        else:
            raise e

mean_absolute_error = float(sum(abs_errors)) / len(abs_errors)
root_mean_squared_error = (sum(squared_errors) / len(squared_errors)) ** 0.5
print("MAE:", mean_absolute_error)
print("RMSE:", root_mean_squared_error)

print("# of matches: ", len(abs_errors))

unique_lakeids = list(set(lakeids))

print("Unique lakeids: ", len(unique_lakeids))

plt.figure(figsize=(10, 6))
plt.hist(abs_errors, 50)
plt.ylabel("Frequency")
plt.xlabel("Absolute Error (µg/L)")
plt.xticks(np.arange(0, 51, 5.0))
plt.title(
    f"Absolute Error with 2019-2024 August Predictions Compared to {"Testing Portion of" if USE_TEST_DATASET_FOR_ERROR_ANALYSIS else "All"} Corresponding In-Situ Data"
)
plt.show()
