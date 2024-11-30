import db_utils
import raster_utils

spatial_prediction_id = input("Enter a spatial prediciton map record's id: ")

downloaded_raster_bytes = db_utils.download_prediction_image_by_record_id(
    spatial_prediction_id
)

max_val = raster_utils.get_max_from_predictions_raster_bytes(downloaded_raster_bytes)
print(f"Max from {spatial_prediction_id}: {max_val}µg/L")
