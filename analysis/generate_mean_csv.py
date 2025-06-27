## Generate mean csv of ALL predictions

import db_utils
import raster_utils
import time
from pprint import pprint
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import os
import json
import pandas as pd
from datetime import datetime
from is_lake_insitu import is_lake_row_insitu
import numpy as np
import joblib

TIF_OUT_FILEPATH = os.getenv("TIF_OUT_FILEPATH")  # for accessing files manually
SAVED_PLOTS_FOLDER_PATH = os.getenv("SAVED_PLOTS_FOLDER_PATH")
ACCESS_STORAGE_MODE = (
    "local"  # "web" | "local" # Web DB OR Copy of Web DB cloned to local computer
)
USE_CACHED_MEANS = (os.getenv("USE_CACHED_MEANS") or "").lower() == "true"
CACHED_MEANS_SAVE_FILE = "summer_means_df.joblib"

LANDSAT_SESSION_UUIDS = os.getenv("LANDSAT_SESSION_UUIDS").split(",")
SENTINEL_SESSION_UUIDS = os.getenv("SENTINEL_SESSION_UUIDS").split(",")

if not os.path.exists(SAVED_PLOTS_FOLDER_PATH):
    os.makedirs(SAVED_PLOTS_FOLDER_PATH)

all_spatial_predictions_list = list(db_utils.spatial_predictions_collection.find({}))
print("number of spatial_predictions: ", len(all_spatial_predictions_list))

if USE_CACHED_MEANS and os.path.exists(CACHED_MEANS_SAVE_FILE):
    print("Using cached spatial prediction means...")
    predictions_df = joblib.load(CACHED_MEANS_SAVE_FILE)
else:
    print("Generating spatial prediction means from rasters on disk...")
    for index in tqdm(range(len(all_spatial_predictions_list))):
        spatial_prediction = all_spatial_predictions_list[index]
        if ACCESS_STORAGE_MODE == "local":
            file_path = os.path.join(
                TIF_OUT_FILEPATH, f"tif_out_{spatial_prediction["session_uuid"]}", spatial_prediction["raster_image"]
            )
            try:
                results_array = raster_utils.get_analytics_from_predictions_raster_file(
                    file_path
                )  # max_val, mean_val, stdev
            except Exception as e:
                print(e)
                continue
        elif ACCESS_STORAGE_MODE == "web":
            raise Exception("Can't use web access mode, not implemented yet")
        else:
            raise Exception('ACCESS_STORAGE_MODE must be either "local" or "web"')

        spatial_prediction["max"] = results_array[0]
        spatial_prediction["min"] = results_array[1]
        spatial_prediction["mean"] = results_array[2]
        spatial_prediction["std"] = results_array[3]

        if spatial_prediction["session_uuid"] in LANDSAT_SESSION_UUIDS:
            spatial_prediction["satellite"] = "landsat8/9"
        elif spatial_prediction["session_uuid"] in SENTINEL_SESSION_UUIDS:
            spatial_prediction["satellite"] = "sentinel2a/b"
        else:
            spatial_prediction["satellite"] = "unknown"


    predictions_df = pd.DataFrame.from_records(all_spatial_predictions_list)
    predictions_df = predictions_df[
        ["lagoslakeid", "date", "max", "min", "mean", "std", "satellite"]
    ]  # Restrict predictions_df to reduce file size

    predictions_df["insitu"] = predictions_df.apply(is_lake_row_insitu, axis=1)
    joblib.dump(predictions_df, CACHED_MEANS_SAVE_FILE)

predictions_df['date'] = pd.to_datetime(predictions_df['date']) # ensure it is a date

predictions_df.to_csv(
    "summer_means.csv", date_format=f"%Y%m%d", float_format="%f", index=False
)  # note %f defaults to 6 digits of precision (won't do crazy scientific notation as str() does)


def get_mean_for_range(df, start_date: datetime, end_date: datetime):
    df_new = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    if len(df_new) == 0:
        print(
            f"Cannot get mean for from {start_date} to {end_date} because there are zero predictions for that range."
        )
    return df_new["mean"].mean(axis=0)  # axis = 0 for columnwise mean


plt.figure("Mean Chl-a Concentration Of All Lakes Over Time", figsize=(18, 9))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(r"%Y-%m-%d"))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))

START_YEAR = 2013
END_YEAR = 2024

for year in range(START_YEAR, END_YEAR + 1):
    dates_to_plot = []
    means_to_plot = []
    date_range = list(
        pd.date_range(
            start=f"{year}-03-01", end=f"{year}-11-01", freq="MS", inclusive="right"
        )
    )  # includes dates from march all the way to oct 31 (excludes jan, feb, nov, dec)

    for i in range(len(date_range) - 1):
        start_date = date_range[i]
        end_date = date_range[i + 1]
        mean = get_mean_for_range(predictions_df, start_date, end_date)
        print(
            f"Mean from {start_date} to {end_date}: ",
            mean,
        )

        dates_to_plot.append(start_date + ((end_date - start_date) / 2))  # middle date
        means_to_plot.append(mean)

    # Shade section of year we are interested in
    plt.axvspan(
        datetime(year, 3, 1),
        datetime(year, 11, 1),
        facecolor="0.5",
        alpha=0.5,
    )

    # Draw midlineof year (July 2nd)
    # plt.axvline(x=datetime(year, 7, 2), color="r")

    plt.plot(
        dates_to_plot, means_to_plot, label=f"{year}"
    )  # plot inside loop so that line is NOT continous from nov to feb (excluded cuz ice)

plt.gcf().autofmt_xdate()  # must be called AFTER plotting

plt.title("Mean Chl-a Concentration Of All Lakes Over Time")
plt.xlabel("Date")
plt.ylabel("Chl-a (µg/L)")
plt.legend()
plt.savefig(
    os.path.join(SAVED_PLOTS_FOLDER_PATH, "mean_of_rasters_across_years.png"),
    bbox_inches="tight",
)
plt.show()

print(
    "Min prediciton_STDEV: ", np.min(predictions_df["std"])
)  # Useful for debugging if there are low stds (caused by errenous or blank images)
print(
    "Max prediciton_STDEV: ", np.max(predictions_df["std"])
)  # Useful for debugging if there are low stds (caused by errenous or blank images)
print("Avg prediciton_STDEV: ", np.mean(predictions_df["std"]))

print("Min prediciton_min: ", np.min(predictions_df["min"]))
print("Max prediciton_min: ", np.max(predictions_df["max"]))
print("Avg prediciton_min: ", np.mean(predictions_df["min"]))

# Inspect very low STD (aka uniform prediction) lakes
predictions_df["is_low_std"] = predictions_df.apply(
    lambda row: row["std"] < 0.000003, axis=1
)
# predictions_df.to_csv(
#     "summer_means_inspect.csv", date_format=f"%Y%m%d", float_format="%f", index=False
# )  # note %f defaults to 6 digits of precision (won't do crazy scientific notation as str() does)

sus_preds = len(predictions_df[predictions_df["is_low_std"] == True])
insitu_sus_preds = len(
    predictions_df[
        (predictions_df["is_low_std"] == True) & (predictions_df["insitu"] == True)
    ]
)
non_insitu_sus_preds = len(
    predictions_df[
        (predictions_df["is_low_std"] == True) & (predictions_df["insitu"] == False)
    ]
)
print("# of predictions with suspiciously low standard deviations: ", sus_preds)
print(
    "# of predictions with suspiciously low standard deviations INSITU: ",
    insitu_sus_preds,
)
print(
    "# of predictions with suspiciously low standard deviations NOT INSITU: ",
    non_insitu_sus_preds,
)

insitu_preds = len(predictions_df[predictions_df["insitu"] == True])
non_insitu_preds = len(predictions_df[predictions_df["insitu"] == False])
print("# of predictions insitu: ", insitu_preds)
print("# of predictions not insitu: ", non_insitu_preds)

ratio_sus_insitu_to_total_insitu_preds = insitu_sus_preds / insitu_preds
ratio_sus_non_insitu_to_total_non_insitu_preds = non_insitu_sus_preds / non_insitu_preds

print(
    "Ratio of suspiciously low standard deviation lake-with-insitu-available predictions to total lake-with-insitu-available predictions: ",
    ratio_sus_insitu_to_total_insitu_preds,
)
print(
    "Ratio of suspiciously low standard deviation lake-without-insitu-available predictions to total lake-without-insitu-available predictions: ",
    ratio_sus_non_insitu_to_total_non_insitu_preds,
)
