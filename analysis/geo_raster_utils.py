import rasterio
from shapely.geometry import Point
import rasterio.mask
import numpy as np
import sys


def get_statistics_abt_point(raster_path: str, lat: float, lng: float):
    with rasterio.open(raster_path) as src:
        x_res = src.res[0]  # same as src.res[1]
        circle = Point(lng, lat).buffer(
            x_res * (60 / float(src.tags()["scale"]))
        )  # however many x_res sized pixels needed for 60m buffer at downloaded scale

        out_image, transformed = rasterio.mask.mask(
            src, [circle], invert=False, crop=True
        )  # read pixels within just the 60m circle mask

    flatten_raster_array = out_image.flatten()
    flatten_raster_array = flatten_raster_array[
        np.isfinite(flatten_raster_array)
    ]  # Remove infinities, nans

    max_val = np.nanmax(flatten_raster_array)
    min_val = np.nanmin(flatten_raster_array)
    mean_val = np.nanmean(flatten_raster_array)  # mean EXCLUDING nans
    stdev = np.nanstd(flatten_raster_array)  # std EXCLUDING nans

    return max_val, min_val, mean_val, stdev


if __name__ == "__main__":
    ilat = 42.20966772115973
    ilng = -79.44488525390626  # Open the raster

    stats = get_statistics_abt_point(sys.argv[1], ilat, ilng)
    print(stats)
