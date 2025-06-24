import os
import rasterio
import numpy as np
from tqdm import tqdm
from pprint import pprint

def is_list_valid(filepath: str):
    try:
        with rasterio.open(filepath) as src:
            # Read the entire image into a numpy array (bands, height, width)
            img = src.read()

            flattened_img = img.flatten()

            if np.all(flattened_img == np.float32("-inf")):
                return False
            return True
    except rasterio.errors.RasterioIOError as e: # Weird file early term error
        return False
    
    return invalid_paths

invalid_paths = []

satellite_tif_folder = input("Enter the path to the downloaded tifs you want to inspect: ")

paths = os.listdir(satellite_tif_folder)

for path in tqdm(paths):
    filepath = os.path.join(satellite_tif_folder, path)
    if not is_list_valid(filepath):
        invalid_paths.append(filepath)

pprint(invalid_paths)
print("# of invalid paths: ", len(invalid_paths))
