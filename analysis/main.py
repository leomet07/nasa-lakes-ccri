import db_utils
import raster_utils
import time

spatial_prediction_id = input("Enter a spatial prediciton map record's id: ")

downloaded_raster_bytes = db_utils.download_prediction_image_by_record_id(
    spatial_prediction_id
)

max_val, mean_val, stdev = raster_utils.get_max_from_predictions_raster_bytes(
    downloaded_raster_bytes
)
print(f"Max from {spatial_prediction_id}: {max_val}µg/L")
print(f"Mean from {spatial_prediction_id}: {mean_val}µg/L")
print(f"STDEV from {spatial_prediction_id}: {stdev}")


# def test():
#     downloaded_raster_bytes, return_date = (
#         db_utils.download_first_prediction_image_by_date_range(
#             100965, "2017-05-31", "2020-05-31"
#         )
#     )

#     max_val = raster_utils.get_max_from_predictions_raster_bytes(
#         downloaded_raster_bytes
#     )
#     print(f"Max from {return_date}: {max_val}µg/L")


# start_time = time.time()

# test()

# end_time = time.time()

# elapsed_time = end_time - start_time

# print(elapsed_time)


# wfwglyjn82le9z5
