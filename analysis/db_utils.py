import os
from dotenv import load_dotenv
from pocketbase import PocketBase  # Client also works the same
import httpx
import rasterio
from datetime import datetime, timezone
from pprint import pprint
from pymongo import MongoClient

load_dotenv()

mongo_client = MongoClient(os.getenv("MONGO_CONNECTION_URI"))
prod_db = mongo_client["prod"]
lakes_collection = prod_db.lakes
spatial_predictions_collection = prod_db.spatial_predictions


def get_prediction_records_by_date_range(
    lagoslakeid: int, start_date: str, end_date: str
):
    # Do filtering database side to avoid making excessive requests here
    filter_str = f'lagoslakeid={lagoslakeid} && date <= "{end_date} 23:59:59.999Z" && date >= "{start_date} 00:00:00.000Z"'
    return client.collection("spatialPredictionMaps").get_full_list(
        query_params={
            "filter": filter_str,
            "sort": "+date",  # Ascending order by date
        }
    )
