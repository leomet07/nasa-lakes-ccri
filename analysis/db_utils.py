import os
from dotenv import load_dotenv
from pocketbase import PocketBase  # Client also works the same
import httpx
import rasterio
from datetime import datetime, timezone
from pprint import pprint
import pymongo
from pprint import pprint

load_dotenv()

mongo_client = pymongo.MongoClient(os.getenv("MONGO_CONNECTION_URI"))
prod_db = mongo_client["prod"]
lakes_collection = prod_db.lakes
spatial_predictions_collection = prod_db.spatial_predictions


def get_prediction_records_by_date_range(
    lagoslakeid: int, start_date: datetime, end_date: datetime
):
    return list(
        spatial_predictions_collection.find(
            {
                "lagoslakeid": lagoslakeid,
                "date": {"$gte": start_date, "$lte": end_date},
            }
        ).sort([("date", pymongo.ASCENDING)])
    )


if __name__ == "__main__":
    pprint(
        get_prediction_records_by_date_range(
            81353, datetime(2024, 4, 1), datetime(2024, 5, 1)
        )
    )
