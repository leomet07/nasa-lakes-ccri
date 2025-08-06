import db_utils
from pprint import pprint
from tqdm import tqdm
from datetime import datetime
import pymongo
from dotenv import load_dotenv
import os

load_dotenv()

SENTINEL_SESSION_UUIDS = os.getenv("SENTINEL_SESSION_UUIDS").split(",")


old_client = pymongo.MongoClient(os.getenv("OLD_MONGO_CONNECTION_URI"))
old_prod_db = old_client["prod"]
old_lakes_collection = old_prod_db.lakes
old_spatial_predictions_collection = old_prod_db.spatial_predictions

print(
    "Number of total preds: ",
    old_spatial_predictions_collection.estimated_document_count(),
)


all_old_spatial_predictions_list = list(
    old_spatial_predictions_collection.find(
        {"session_uuid": {"$in": SENTINEL_SESSION_UUIDS}}
    )
)  # query all then filter, faster than searching and querying on db level

print("Number of sentinel preds:", len(all_old_spatial_predictions_list))

for old_pred in all_old_spatial_predictions_list:
    # mutate old predictions
    old_pred["scale"] = int(old_pred["scale"])
    old_pred["satellite"] = "sentinel2a/b"


# insert all sentinel preds into new db
db_utils.spatial_predictions_collection.insert_many(all_old_spatial_predictions_list)
