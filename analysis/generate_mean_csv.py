## Generate mean csv of ALL predictions

import db_utils
import raster_utils
import time
from pprint import pprint
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
import json
import pandas as pd

ROOT_DB_FILEPATH = os.getenv("ROOT_DB_FILEPATH") # for accessing files manually
ACCESS_STORAGE_MODE = "local" # "web" | "local" # Web DB OR Copy of Web DB cloned to local computer

all_spatial_predictions = db_utils.client.collection("spatialPredictionMaps").get_full_list(batch=100_000)

print("number of spatial_predictions: ", len(all_spatial_predictions))

all_spatial_predictions_list = list(map(vars, all_spatial_predictions))

for index in tqdm(range(len(all_spatial_predictions_list))):
    spatial_prediction = all_spatial_predictions_list[index]
    if ACCESS_STORAGE_MODE == "local":
        file_path = f"{ROOT_DB_FILEPATH}/pb_data/storage/{spatial_prediction["collection_id"]}/{spatial_prediction["id"]}/{spatial_prediction["raster_image"]}"
        results_array = raster_utils.get_max_from_predictions_raster_file(
            file_path
        )  # max_val, mean_val, stdev
    elif ACCESS_STORAGE_MODE == "web": 
        results_array = raster_utils.get_max_from_predictions_raster_bytes(
            db_utils.download_prediction_image_by_record_id(spatial_prediction["id"])
        )  # max_val, mean_val, stdev
    else:
        raise Exception('ACCESS_STORAGE_MODE must be either "local" or "web"')
    
    spatial_prediction["max"] = str(results_array[0])
    spatial_prediction["mean"] = str(results_array[1])
    spatial_prediction["std"] = str(results_array[2])

df = pd.DataFrame.from_records(all_spatial_predictions_list)

df.to_csv("summer_means_full.csv")
# Restrict DF to reduce file size
df = df[["lagoslakeid", "date", "max", "mean", "std"]]
df.to_csv("summer_means.csv")

with open("all_predictions_dump.json", "w") as json_file:
    json_file.write(json.dumps(all_spatial_predictions_list, indent=4))