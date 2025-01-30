# Setup dotenv
from dotenv import load_dotenv
import os

load_dotenv()

from pocketbase import PocketBase  # Client also works the same
from pocketbase.client import FileUpload
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import time
from datetime import datetime
from pathlib import Path
import json
from pyproj import Proj
import uuid
from tqdm import tqdm
import gc
import model_training


client = PocketBase(os.getenv("PUBLIC_POCKETBASE_URL"))
admin_data = client.admins.auth_with_password(os.getenv("POCKETBASE_ADMIN_EMAIL"), os.getenv("POCKETBASE_ADMIN_PASSWORD"))


session_uuid = str(uuid.uuid4())
print("Current session id: ", session_uuid)


input_tif_folder = "all_lakes_all_images_2025_01_28" # Specify the folder inside of the tar
paths = os.listdir(input_tif_folder)
print("Number of files to run: ", len(paths))

png_out_folder = os.path.join("all_png_out", f"png_out_{session_uuid}")
if not os.path.exists(png_out_folder):
    os.makedirs(png_out_folder)
    
tif_out_folder = os.path.join("all_tif_out", f"tif_out_{session_uuid}")
if not os.path.exists(tif_out_folder):
    os.makedirs(tif_out_folder)

session_statues_path = "session_statuses/"
if not os.path.exists(session_statues_path):
    os.makedirs(session_statues_path)

error_paths = []


def add_suffix_to_filename_at_tif_path(filename : str, suffix : str):
    parts = filename.split(".")
    newfilename =  parts[0] + f"_{suffix}." + ".".join(parts[1:])

    to_tif_folder_path = os.path.join(tif_out_folder, newfilename.split("/")[-1])

    return to_tif_folder_path


def modify_tif(input_tif : str, SA_constant : float, Max_depth_constant : float, pct_dev_constant: float, pct_ag_constant : float) -> str:
    with rasterio.open(input_tif) as src:
        raster_data = src.read()
        profile = src.profile  # Get the profile of the existing raster


    # create new bands
    SA_band = np.full_like(raster_data[0], SA_constant, dtype=raster_data.dtype)
    Max_depth_band = np.full_like(raster_data[0], Max_depth_constant, dtype=raster_data.dtype)
    pct_dev_band = np.full_like(raster_data[0], pct_dev_constant, dtype=raster_data.dtype)
    pct_ag_band = np.full_like(raster_data[0], pct_ag_constant, dtype=raster_data.dtype)

    # update profile to reflect additional bands
    profile.update(count=13)  # (update count to include original bands + 4 new bands)

    # output GeoTIFF file
    modified_tif = add_suffix_to_filename_at_tif_path(input_tif, "modified")

    # write the modified raster data to the new GeoTIFF file
    with rasterio.open(modified_tif, 'w', **profile) as dst:
        # write original bands
        for i in range(1, raster_data.shape[0] + 1):
            dst.write(raster_data[i-1], indexes=i)

        # write additional bands
        dst.write(SA_band, indexes=raster_data.shape[0] + 1)
        dst.write(Max_depth_band, indexes=raster_data.shape[0] + 2)
        dst.write(pct_dev_band, indexes=raster_data.shape[0] + 3)
        dst.write(pct_ag_band, indexes=raster_data.shape[0] + 4)


        dst.transform = src.transform
        dst.crs = src.crs

    # print(f"Created {modified_tif} with the four extra bands data from constants")
    return modified_tif


def predict(input_tif : str, id: int, display = True):
    modified_tif = add_suffix_to_filename_at_tif_path(input_tif, "modified")
    with rasterio.open(modified_tif) as src:
        raster_data = src.read()
        profile = src.profile
        transform = src.transform
        crs = src.crs

    # reshape raster data to 2D array for model prediction
    n_bands, n_rows, n_cols = raster_data.shape
    raster_data_2d = raster_data.reshape(n_bands, -1).T

    # handle NaN values by replacing them with the mean of each band
    nan_mask = np.isnan(raster_data_2d)
    means = np.nanmean(raster_data_2d, axis=0)
    raster_data_2d[nan_mask] = np.take(means, np.where(nan_mask)[1])

    # perform the prediction
    predictions = model_training.andrew_model.predict(raster_data_2d)

    # reshape the predictions back to the original raster shape
    predictions_raster = predictions.reshape(n_rows, n_cols)

    # save the prediction result as a new raster file
    output_tif = add_suffix_to_filename_at_tif_path(input_tif, "predicted")

    with rasterio.open(output_tif, 'w',
                      driver='GTiff',
                      height=n_rows,
                      width=n_cols,
                      count=1,
                      dtype=predictions_raster.dtype,
                      crs=crs,
                      transform=transform) as dst:
        dst.write(predictions_raster, 1)

    # plot the result
    if display:
        plt.imshow(predictions_raster, cmap='viridis')
        plt.colorbar()
        plt.title('Predicted Values')
        plt.show()

    return output_tif, predictions_raster


def save_png(input_tif, out_folder, predictions_raster, date, scale, display=True):
    first_pixel_value = predictions_raster[0, 0]
    masked_raster = np.where(predictions_raster == first_pixel_value, np.nan, predictions_raster)

    min_value = 0
    max_value = 60
    increment = 5

    fig = plt.figure(figsize=(10, 8))
    plt.imshow(masked_raster, cmap='viridis', interpolation='none', vmin=min_value, vmax=max_value)
    plt.axis('off')
    stem = Path(input_tif).stem

    # values = np.arange(min_value, max_value + increment, increment)
    # cbar = plt.colorbar()
    # cbar.set_label(f'Predicted chlorophyll-A in ug/L \n ({date}, scale: {scale})')
    # cbar.set_ticks(values)
    # cbar.set_ticklabels([str(val) for val in values])

    # png filename
    output_png = stem + ".png"
    output_png_path = os.path.join(out_folder, output_png)
    # save the png
    plt.savefig(output_png_path, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    if display:
        plt.show()
    else:
        plt.close(fig)
    return output_png_path


def upload_spatial_map(lakeid : int, raster_image_path: str, display_image_path : str, datestr : str, corners : list, scale : int):
    # Step 1: Find Lakeid
    filter_str = f"lagoslakeid='{lakeid}'"

    matched_lake = client.collection("lakes").get_list(1, 20, {"filter": filter_str}).items[0]
    lake_db_id = matched_lake.id

    # print("Found lake's id in the DB: ", lake_db_id)

    dateiso = datetime.strptime(datestr, '%Y-%m-%d').isoformat()
    # Step 2

    body = {
        "raster_image" : None,
        "display_image" : None,
        "date" : dateiso, # utc
        "corner1latitude": corners[0][0],
        "corner1longitude": corners[0][1],
        "corner2latitude": corners[1][0],
        "corner2longitude": corners[1][1],
        "scale" : scale,
        "session_uuid" : session_uuid,
        "lake" : lake_db_id,
        "lagoslakeid" : lakeid
    }

    # print("Inside upload:" , body)
    with open(raster_image_path, "rb") as rasterimage:
        body["raster_image"] = FileUpload((f"raster_image_{lakeid}_{datestr}.tif", rasterimage))
        with open(display_image_path, "rb") as displayimage:
            body["display_image"] = FileUpload((f"display_image_{lakeid}_{datestr}.png", displayimage))
            created = client.collection("spatialPredictionMaps").create(body)

    lake_result = client.collection("lakes").update(lake_db_id, {"spatialPredictions+" : created.id}) # here the library does not snakeCase

    # print("Lake result: ",lake_result.__dict__)


for path_tif in tqdm(paths):
    path_tif = os.path.join(input_tif_folder, path_tif)
    try:
        with rasterio.open(path_tif) as raster:
            tags = raster.tags()
            id = int(tags["id"])
            date = tags["date"] # date does NOT do anything here, just for title
            scale = tags["scale"] # scale does NOT do anything here, just for title

            top_left = raster.transform * (0, 0)
            bottom_right = raster.transform * (raster.width, raster.height)
            crs = raster.crs

        p = Proj(crs)
        # Output is in the format: (lat, long)
        corner1 = list(p(top_left[0], top_left[1], inverse=True)[::-1])
        corner2 = list(p(bottom_right[0], bottom_right[1], inverse=True)[::-1])
        corners = [corner1, corner2]
        # print("id: ", id, " date: ", date, " scale: ", scale, " corners: ", corners)

        # Get constants
        SA_constant, Max_depth_constant, pct_dev_constant, pct_ag_constant = model_training.get_constants(id)
        # print("Constants based on id: ", SA_constant, Max_depth_constant, pct_dev_constant, pct_ag_constant)


        modified_path_tif = modify_tif(path_tif, SA_constant, Max_depth_constant, pct_dev_constant, pct_ag_constant)

        output_tif, predictions_loop = predict(path_tif, id, display = False)

        output_path_png = save_png(path_tif, png_out_folder, predictions_loop, date, scale, display = False)

        upload_spatial_map(id, output_tif, output_path_png, date, corners, scale)

        with open(os.path.join(session_statues_path, f"successes_{session_uuid}.status.txt"), "a") as file_obj:
            file_obj.write(path_tif +"\n")

        gc.collect() # Clear memory after prediction
    except Exception as e:
        print("Error: ", e)
        error_paths.append(path_tif)

print(f"Successfully finished {len(paths)} uploads with {len(error_paths)} errors")
print("Session ID: ", session_uuid)

with open(os.path.join(session_statues_path, f"error_paths_{session_uuid}.json"), "w") as file:
    file.write(json.dumps(error_paths))
