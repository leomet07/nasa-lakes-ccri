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
from is_lake_insitu import is_lake_row_insitu
import numpy as np
import rasterio

ROOT_DB_FILEPATH = os.getenv("ROOT_DB_FILEPATH") # for accessing files manually
ACCESS_STORAGE_MODE = "local" # "web" | "local" # Web DB OR Copy of Web DB cloned to local computer

all_spatial_predictions = db_utils.client.collection("spatialPredictionMaps").get_full_list(batch=100_000)

print("number of spatial_predictions: ", len(all_spatial_predictions))

all_spatial_predictions_list = list(map(vars, all_spatial_predictions))

vals = []

for index in tqdm(range(len(all_spatial_predictions_list))):
    spatial_prediction = all_spatial_predictions_list[index]
    if ACCESS_STORAGE_MODE == "local":
        file_path = f"{ROOT_DB_FILEPATH}/pb_data/storage/{spatial_prediction["collection_id"]}/{spatial_prediction["id"]}/{spatial_prediction["raster_image"]}"
        raster_array = raster_utils.get_raster_array_from_file(file_path)
    elif ACCESS_STORAGE_MODE == "web": 
        downloaded_raster_bytes = db_utils.download_prediction_image_by_record_id(spatial_prediction["id"])
        raster_array = raster_utils.get_raster_array_from_bytes(downloaded_raster_bytes)
    else:
        raise Exception('ACCESS_STORAGE_MODE must be either "local" or "web"')

    flatened_raster_array = raster_array.flatten()
    # print("# of pixels total: ", len(flatened_raster_array))
    no_nans = flatened_raster_array[~np.isnan(flatened_raster_array)]
    # print("# of pixels no_nans: ", len(no_nans))

    vals.extend(no_nans)

print("# of vals: ", len(vals))

plt.hist(vals, 200)
plt.xlim(0, 75)
plt.xlabel("Predicted chl-a (ug/l)")
plt.ylabel("Frequency")
plt.title("Real/Production Prediction Histogram")
plt.show()