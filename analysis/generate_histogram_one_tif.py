## Generate a histogram of one chla prediction/tif
import raster_utils
from matplotlib import pyplot as plt
import numpy as np

file_path = input("Enter the filepath to the tif you want to analyze: ")

raster_array = raster_utils.get_raster_array_from_file(file_path)

flattened_raster_array = raster_array.flatten()

print(flattened_raster_array)

print("# of pixels total: ", len(flattened_raster_array))
no_nans = flattened_raster_array[~np.isnan(flattened_raster_array)]
print("# of pixels no_nans: ", len(no_nans))

vals = no_nans # Used to be array for when multiple predictions, but this is just one
print("# of vals: ", len(vals))

plt.hist(vals, 200)
plt.xlim(0, 75)
plt.xlabel("Predicted chl-a (ug/l)")
plt.ylabel("Frequency")
plt.title(f"{file_path} Prediction Histogram")
plt.show()