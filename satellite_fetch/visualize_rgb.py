#visualize rgb image when only 3 bands are present (b2, b3, b4 in that order)
from matplotlib import pyplot as plt
import rasterio

import numpy as np
out_file = input("Enter the path of a tif to inspect RGB: ")

with rasterio.open(out_file) as src:
    # Read the number of bands and the dimensions
    num_bands = src.count
    height = src.height
    width = src.width
    print(src.profile)

    print(f"Number of bands: {num_bands}")
    print(f"Dimensions: {width} x {height}")

    # Read the entire image into a numpy array (bands, height, width)
    img = src.read()

# Replace img nans with zero
img[~np.isfinite(img)] = 0

red_band = img[2, :, :] # B4
green_band = img[1, :, :] # B3
blue_band = img[0, :, :]# B2

# Normalize bands to 0-1 range
red_band = (red_band - np.min(red_band)) / (np.max(red_band) - np.min(red_band))
green_band = (green_band - np.min(green_band)) / (np.max(green_band) - np.min(green_band))
blue_band = (blue_band - np.min(blue_band)) / (np.max(blue_band) - np.min(blue_band))

# Stack the bands into a 3D array
rgb_image = np.dstack((red_band, green_band, blue_band))

# Display the RGB image
plt.imshow(rgb_image)
plt.show()