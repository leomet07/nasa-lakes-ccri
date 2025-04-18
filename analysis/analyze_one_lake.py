import db_utils
import raster_utils
import time
from pprint import pprint
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import os
import sys
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

if len(sys.argv) < 2:
    raise Exception(
        "You must specify the lagoslakeid as a command line argument to analyze it."
    )

lagoslakeid = int(sys.argv[1])
matched_lake = db_utils.lakes_collection.find_one({"lagoslakeid": lagoslakeid})

if not matched_lake:
    raise Exception(f"Lake with lagoslakeid = {lagoslakeid} was not found.")

lake_spatial_predictions_list = list(
    db_utils.spatial_predictions_collection.find(
        {"lagoslakeid": lagoslakeid},
    )
)
print("number of spatial_predictions: ", len(lake_spatial_predictions_list))


print("Generating spatial prediction means from rasters on disk...")

dates_to_plot = []
y_s_to_plot = []

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(r"%Y-%m-%d"))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))


for index in tqdm(range(len(lake_spatial_predictions_list))):
    spatial_prediction = lake_spatial_predictions_list[index]
    if ACCESS_STORAGE_MODE == "local":
        file_path = os.path.join(TIF_OUT_FILEPATH, spatial_prediction["raster_image"])
        results_array = raster_utils.get_analytics_from_predictions_raster_file(
            file_path
        )  # max_val, mean_val, stdev
    elif ACCESS_STORAGE_MODE == "web":
        raise Exception("Can't use web access mode, not implemented yet")
    else:
        raise Exception('ACCESS_STORAGE_MODE must be either "local" or "web"')

    spatial_prediction["max"] = results_array[0]
    spatial_prediction["min"] = results_array[1]
    spatial_prediction["mean"] = results_array[2]
    spatial_prediction["std"] = results_array[3]

    dates_to_plot.append(spatial_prediction["date"])
    y_s_to_plot.append(spatial_prediction["mean"])

plt.scatter(
    dates_to_plot, y_s_to_plot
)  # plot inside loop so that line is NOT continous from nov to feb (excluded cuz ice)
# TODO: to use .plot, dates must be sorted (in monogdb call use sort={})

plt.gcf().autofmt_xdate()  # must be called AFTER plotting

plt.title(f"(Mean) Chl-a Concentration Of Lake {matched_lake["name"]} Over Time")
plt.xlabel("Date")
plt.ylabel("Chl-a (µg/L)")
plt.show()
