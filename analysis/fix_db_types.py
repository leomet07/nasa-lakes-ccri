import db_utils
from pprint import pprint
from tqdm import tqdm
from datetime import datetime

all_spatial_predictions_list = list(
    db_utils.spatial_predictions_collection.find({})
)  # query all then filter, faster than searching and querying on db level

for spatial_prediction in tqdm(all_spatial_predictions_list):
    date_obj = datetime.fromisoformat(spatial_prediction["date"])
    scale = int(spatial_prediction["scale"])

    db_utils.spatial_predictions_collection.update_one(
        {"_id": spatial_prediction["_id"]},
        {"$set": {"date": date_obj, "satellite": "landsat8/9", "scale": scale}},
        upsert=False,
    )
