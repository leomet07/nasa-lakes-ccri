import rasterio
from pyproj import CRS, Transformer, Proj

input_tif = "S2_Chautauqua_v2_predictions.tif"

with rasterio.open(input_tif) as src:
    top_left = src.transform * (0, 0)
    bottom_right = src.transform * (src.width, src.height)
    ## None of the commented stuff was a perfect overlay on the leaflet map
    ## It was always off by a 50 or so meters
    crs = src.crs
    """
    bounds = [
        top_left[0],
        top_left[1],
        bottom_right[0],
        bottom_right[1],
    ]  # NOT [[],[]], instead [a,b,c,d]
    print("BOUNDS: ", bounds)
    # raster_data = src.read()
    # profile = src.profile
    # transform = src.transform
    # print(crs)
    # cords = src.xy(transform, 0, 0)

    xmin, ymin, xmax, ymax = transform_bounds(crs.to_epsg(), 4326, *src.bounds)

    print([xmin, ymin, xmax, ymax])

    top_left_coords = [round(ymin, 7), round(xmax, 7)]
    bottom_right_coords = [round(ymax, 7), round(xmin, 7)]
    print(bottom_right_coords)
    print(top_left_coords)
print("\n\nErin\n\n")

import rasterio


input_tif = "S2_Chautauqua_v2_predictions.tif"

with rasterio.open(input_tif) as src:
    crs = src.crs
    bounds = src.bounds
    original_crs = CRS.from_wkt(crs.to_wkt())
    original_epsg = original_crs.to_epsg()
    print(f"Original EPSG Code: {original_epsg}")
    target_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(
        crs, target_crs, always_xy=True, allow_ballpark=False
    )
    min_lon, min_lat = transformer.transform(bounds.left, bounds.bottom)
    max_lon, max_lat = transformer.transform(bounds.right, bounds.top)
    lat_lon_bounds = [[min_lat, min_lon], [max_lat, max_lon]]
    print("Bounds in latitude and longitude:", lat_lon_bounds)
"""
p = Proj(crs)
print(p(top_left[0], top_left[1], inverse=True)[::-1])
print(p(bottom_right[0], bottom_right[1], inverse=True)[::-1])
