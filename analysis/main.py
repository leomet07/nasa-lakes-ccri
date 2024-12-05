import db_utils
import raster_utils
import time
from pprint import pprint
from tqdm import tqdm
from matplotlib import pyplot as plt
from datetime import datetime
import os

OUT_DIRECTORY = "saved_plots"
if not os.path.exists(OUT_DIRECTORY):
    os.mkdir(OUT_DIRECTORY)


def plot_predictions_for_one_year(lagoslakeid: int, year: int):
    spatial_predictions = db_utils.get_prediction_records_by_date_range(
        lagoslakeid, f"{year}-01-01", f"{year}-12-31"
    )

    print("# of predictions: ", len(spatial_predictions))

    x = []
    y = []

    for spatial_prediction in tqdm(spatial_predictions):
        results_array = raster_utils.get_max_from_predictions_raster_bytes(
            db_utils.download_prediction_image_by_record_id(spatial_prediction.id)
        )  # max_val, mean_val, stdev

        date_obj = datetime.fromisoformat(spatial_prediction.date)
        # date_obj = date_obj.replace(year=1971)
        x.append(date_obj)
        y.append(results_array[0])  # y will be maxs

    plt.plot(x, y, label=f"Concentrations in {year}")


LAKE_ID = 81353

for year in range(2019, 2024):  # Exclusive of year
    plot_predictions_for_one_year(LAKE_ID, year)

plt.legend()
plt.title(f"Max Chl-A of Lake{LAKE_ID} vs Time")
plt.xlabel("Time")
plt.ylabel("Chl-a [µg/L]")
plt.savefig(
    os.path.join(OUT_DIRECTORY, f"plot_{str(datetime.now()).replace(" ", "_")}.png")
)
plt.show()
