import numpy as np
import rasterio


def get_max_from_predictions_raster_bytes(raster: bytes) -> float:
    with rasterio.io.MemoryFile(raster) as memfile:
        with memfile.open() as src:
            # Read the number of bands and the dimensions
            num_bands = (
                src.count
            )  # This should be 1 for the output of the prediction file
            if num_bands != 1:
                raise Exception(
                    f"Number of bands in file should be 1 instead of {num_bands}. Are you sure this file is chl_a predictions from the model?"
                )
            height = src.height
            width = src.width

            raster_array = src.read()

            max_val = raster_array.max()  # This DOES work with 2d arrays

            # raster_array = raster_array.flatten()
            # top_ten_highest_indices = np.argpartition(raster_array, -10)[-10:]
            # top_ten = raster_array[top_ten_highest_indices]
            # top_ten.sort()
            # print("Top ten: ", top_ten)

            return max_val


def get_max_from_predictions_raster_file(file_path: str) -> int:
    print(f"Opening {file_path}...")
    with rasterio.open(file_path) as src:
        # Read the number of bands and the dimensions
        num_bands = src.count  # This should be 1 for the output of the prediction file
        if num_bands != 1:
            raise Exception(
                f"Number of bands in {file_path} should be 1 instead of {num_bands}. Are you sure this file is chl_a predictions from the model?"
            )
        height = src.height
        width = src.width

        raster_array = src.read()

        max_val = raster_array.max()  # This DOES work with 2d arrays

        # raster_array = raster_array.flatten()
        # top_ten_highest_indices = np.argpartition(raster_array, -10)[-10:]
        # top_ten = raster_array[top_ten_highest_indices]
        # top_ten.sort()
        # print("Top ten: ", top_ten)

        return max_val


if __name__ == "__main__":

    out_file = input("Enter the path of a PREIDCITION (not input) tif to inspect: ")

    if out_file == "":
        raise Exception("Prediction out file must be provided")

    print(f"Max concentration: {get_max_from_predictions_raster_file(out_file)} µg/L")
