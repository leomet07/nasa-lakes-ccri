import db_utils
import raster_utils
import time
from pprint import pprint
from tqdm import tqdm
from matplotlib import pyplot as plt
from datetime import datetime

spatial_predictions = db_utils.get_prediction_records_by_date_range(
    81353, "2019-01-01", "2019-12-31"
)

print("# of predictions: ", len(spatial_predictions))

x = []
y = []

for spatial_prediction in tqdm(spatial_predictions):
    results_array = raster_utils.get_max_from_predictions_raster_bytes(
        db_utils.download_prediction_image_by_record_id(spatial_prediction.id)
    )  # max_val, mean_val, stdev

    date_obj = datetime.fromisoformat(spatial_prediction.date)
    x.append(date_obj)
    y.append(results_array[1])  # y will be means


plt.plot(x, y, label="2019")
plt.legend()
plt.title("Mean Chl-A vs Time")
plt.xlabel("Time")
plt.ylabel("Chl-a")
plt.show()
