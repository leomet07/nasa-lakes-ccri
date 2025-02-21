import db_utils
import raster_utils
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

load_dotenv()
ROOT_DB_FILEPATH = os.getenv("ROOT_DB_FILEPATH")  # for accessing files manually

all_data = pd.read_csv(
    os.getenv("ALL_INPUT_DATA_CSV")
)  # this later creates cleaned_data for training model

all_data = all_data[all_data["chl_a"] < 100]  # filter to chl_a less than 100

for index, row in tqdm(all_data.iterrows(), total=len(all_data)):
    lagoslakeid = row["lagoslakei"]
    lat = row["MEAN_lat"]
    lng = row["MEAN_long"]
    true_chl_a = row["chl_a"]

    date = row["date"]  # satelltie fly over date ("sample_date" is for sample_date)

    matched_predictions = db_utils.get_prediction_records_by_date_range(
        lagoslakeid, date, date
    )

    if len(matched_predictions) == 0:
        continue  # did not find any prediction image with insitu date

    print(f"Found prediction for Lake{lagoslakeid} on {date}")

    spatial_prediction = vars(matched_predictions[0])
    file_path = f"{ROOT_DB_FILEPATH}/pb_data/storage/{spatial_prediction["collection_id"]}/{spatial_prediction["id"]}/{spatial_prediction["raster_image"]}"

    try:
        stats = raster_utils.get_analytics_from_circular_section_in_raster_file(
            file_path, lat, lng
        )

        prediction_mean_val = stats[2]
        error = prediction_mean_val - true_chl_a

        print(
            "True Value: ",
            true_chl_a,
            "Prediction (mean) value: ",
            prediction_mean_val,
            "Error: ",
            error,
        )
    except ValueError as e:
        # ValueError: zero-size array to reduction operation fmax which has no identity
        # Means only nans or -infs in circle raster which got filtered out
        if str(e).startswith("zero-size array"):
            print(f"Only NAN values within circle about point. Was there a cloud?")
        else:
            raise e
