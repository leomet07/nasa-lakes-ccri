## Generate a histogram of all chla predictions

import db_utils
import raster_utils
import time
from pprint import pprint
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
import json
import pandas as pd
from datetime import datetime
import numpy as np
import rasterio

TIF_OUT_FILEPATH = os.getenv("TIF_OUT_FILEPATH")  # for accessing files manually
SAVED_PLOTS_FOLDER_PATH = os.getenv("SAVED_PLOTS_FOLDER_PATH")
ACCESS_STORAGE_MODE = (
    "local"  # "web" | "local" # Web DB OR Copy of Web DB cloned to local computer
)

all_spatial_predictions_list = list(db_utils.spatial_predictions_collection.find({}))

print("number of spatial_predictions: ", len(all_spatial_predictions_list))

vals = []

for index in tqdm(range(len(all_spatial_predictions_list))):
    spatial_prediction = all_spatial_predictions_list[index]
    if ACCESS_STORAGE_MODE == "local":
        file_path = os.path.join(TIF_OUT_FILEPATH, spatial_prediction["raster_image"])
        raster_array = raster_utils.get_raster_array_from_file(file_path)
    elif ACCESS_STORAGE_MODE == "web":
        raise Exception("Can't use web access mode, not implemented yet")
    else:
        raise Exception('ACCESS_STORAGE_MODE must be either "local" or "web"')

    flatened_raster_array = raster_array.flatten()
    # print("# of pixels total: ", len(flatened_raster_array))
    no_nans = flatened_raster_array[~np.isnan(flatened_raster_array)]
    # print("# of pixels no_nans: ", len(no_nans))

    vals.extend(no_nans)

print("# of vals: ", len(vals))
print("Mean of vals: ", np.mean(vals))
print("Min of vals: ", np.min(vals))
print("Max of vals: ", np.max(vals))

plt.hist(vals, 200)
plt.xlim(0, 75)
plt.xlabel("Predicted chl-a (ug/l)")
plt.ylabel("Frequency")
plt.title("Real/Production Prediction Histogram")
plt.savefig(
    os.path.join(
        SAVED_PLOTS_FOLDER_PATH, f"{str(datetime.now())}-all-pixel-histogram.png"
    )
)
plt.show()
