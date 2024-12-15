import db_utils
import raster_utils
import time
from pprint import pprint
from tqdm import tqdm
from matplotlib import pyplot as plt
from datetime import datetime
import os


## Useful for inspecting "blank" (aka uniform) prediction output
print(raster_utils.get_max_from_predictions_raster_bytes(db_utils.download_prediction_image_by_record_id("y1n180pncty3569")))