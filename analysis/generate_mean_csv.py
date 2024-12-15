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
from datetime import datetime

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
    
    spatial_prediction["max"] = results_array[0]
    spatial_prediction["mean"] = results_array[1]
    spatial_prediction["std"] = results_array[2]
    spatial_prediction["date"] = datetime.fromisoformat(spatial_prediction["date"])

predictions_df = pd.DataFrame.from_records(all_spatial_predictions_list)

predictions_df = predictions_df[["lagoslakeid", "date", "max", "mean", "std"]] # Restrict predictions_df to reduce file size
predictions_df.to_csv("summer_means.csv", date_format=f'%Y%m%d', float_format="%f", index=False) # note %f defaults to 6 digits of precision (won't do crazy scientific notation as str() does)

# Get super means!
print("Total rows: ", len(predictions_df))
def get_august_mean_for_year(df, year: int):
    df_new = df[(df["date"] > f'{year}-07-25') & (df["date"] < f'{year}-09-05')]
    return df_new["mean"].mean(axis=0) # axis = 0 for columnwise mean

for year in range(2019, 2024):
    print(f"Mean for {year}: ", get_august_mean_for_year(predictions_df, year))