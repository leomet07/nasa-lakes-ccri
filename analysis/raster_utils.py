import numpy as np
import rasterio

def check_src(src):
    # Read the number of bands and the dimensions
    num_bands = src.count  # This should be 1 for the output of the prediction file
    if num_bands != 1:
        raise Exception(
            f"Number of bands should be 1 instead of {num_bands}. Are you sure this file is chl_a predictions from the model?"
        )

def run_analytics_on_raster(raster_array):
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


def get_raster_array_from_file(file_path: str):
    with rasterio.open(file_path) as src:
        check_src(src)
        return src.read()

def get_raster_array_from_bytes(raster_bytes: bytes):
    with rasterio.io.MemoryFile(raster_bytes) as memfile:
        with memfile.open() as src:
            check_src(src)
            return src.read()

def get_max_from_predictions_raster_bytes(raster_bytes: bytes) -> float:
    raster_array = get_raster_array_from_bytes(raster_bytes)
    return run_analytics_on_raster(raster_array)


def get_max_from_predictions_raster_file(file_path: str) -> int:
    raster_array = get_raster_array_from_file(file_path)
    return run_analytics_on_raster(raster_array)


if __name__ == "__main__":

    out_file = input("Enter the path of a PREIDCITION (not input) tif to inspect: ")

    if out_file == "":
        raise Exception("Prediction out file must be provided")

    print(f"Max concentration: {get_max_from_predictions_raster_file(out_file)} µg/L")
