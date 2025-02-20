import rasterio
import pyproj
from pyproj import Proj
from shapely.geometry import Point, mapping
import rasterio.mask
import numpy as np
from functools import partial
from shapely.ops import transform

print(
    "This script helps see the geographic coordinates of the top left and bottom right corners of the input tiff"
)
input_tif = input("Enter the filename: ")

lat = 42.20966772115973
lng = -79.44488525390626  # Open the raster

with rasterio.open(input_tif) as src:
    x_res = src.res[0]  # same as src.res[1]
    circle = Point(lng, lat).buffer(
        x_res * (60 / float(src.tags()["scale"]))
    )  # however many x_res sized pixels needed for 60m buffer at downloaded scale

    out_image, transformed = rasterio.mask.mask(src, [circle], invert=False, crop=True)

    out_profile = src.profile.copy()

    print("width, height:", out_image.shape[2], out_image.shape[1])

    print(out_image)

out_profile.update(
    {
        "width": out_image.shape[2],
        "height": out_image.shape[1],
        "transform": transformed,
    }
)
with rasterio.open("out.tif", "w", **out_profile) as dst:
    dst.write(out_image)
