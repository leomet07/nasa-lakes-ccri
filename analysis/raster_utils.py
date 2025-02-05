import numpy as np
import rasterio


def run_analytics_on_opened_raster(src):
    # Assumes src comes from:     with rasterio.open(file_path) as src:

    # Read the number of bands and the dimensions
    num_bands = src.count  # This should be 1 for the output of the prediction file
    if num_bands != 1:
        raise Exception(
            f"Number of bands should be 1 instead of {num_bands}. Are you sure this file is chl_a predictions from the model?"
        )
    height = src.height
    width = src.width

    raster_array = src.read()

    flatten_raster_array = raster_array.flatten()

    max_val = np.nanmax(flatten_raster_array)
    min_val = np.nanmin(flatten_raster_array)

    mean_val = np.nanmean(flatten_raster_array) # mean EXCLUDING nans
    stdev = np.nanstd(flatten_raster_array) # std EXCLUDING nans

    # top_ten_highest_indices = np.argpartition(flatten_raster_array, -10)[-10:]
    # top_ten = flatten_raster_array[top_ten_highest_indices]
    # top_ten.sort()
    # print("Top ten: ", top_ten)

    return max_val, min_val, mean_val, stdev


def get_max_from_predictions_raster_bytes(raster: bytes) -> float:
    with rasterio.io.MemoryFile(raster) as memfile:
        with memfile.open() as src:
            return run_analytics_on_opened_raster(src)


def get_max_from_predictions_raster_file(file_path: str) -> int:
    with rasterio.open(file_path) as src:
        return run_analytics_on_opened_raster(src)


if __name__ == "__main__":

    out_file = input("Enter the path of a PREIDCITION (not input) tif to inspect: ")

    if out_file == "":
        raise Exception("Prediction out file must be provided")

    print(f"Max concentration: {get_max_from_predictions_raster_file(out_file)} µg/L")
