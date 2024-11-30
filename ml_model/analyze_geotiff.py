import rasterio
from pyproj import Proj

print(
    "This script helps see the geographic coordinates of the top left and bottom right corners of the input tiff"
)
input_tif = input("Enter the filename: ") or "S2_Chautauqua_v2_predictions.tif"

with rasterio.open(input_tif) as src:
    tags = src.tags()
    print("Tags: ", tags)
    top_left = src.transform * (0, 0)
    bottom_right = src.transform * (src.width, src.height)
    crs = src.crs

p = Proj(crs)
print("Output is in the format: (lat, long)")
print(p(top_left[0], top_left[1], inverse=True)[::-1])
print(p(bottom_right[0], bottom_right[1], inverse=True)[::-1])
