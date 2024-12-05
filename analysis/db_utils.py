import os
from dotenv import load_dotenv
from pocketbase import PocketBase  # Client also works the same
import httpx
import rasterio
from datetime import datetime, timezone
from pprint import pprint

load_dotenv()

client = PocketBase(os.getenv("PUBLIC_POCKETBASE_URL"))
admin_data = client.admins.auth_with_password(
    os.getenv("POCKETBASE_ADMIN_EMAIL"), os.getenv("POCKETBASE_ADMIN_PASSWORD")
)


def download_raster_image_bytes_from_record(record) -> bytes:
    url = f"{os.getenv("PUBLIC_POCKETBASE_URL")}/api/files/{record.collection_id}/{record.id}/{record.raster_image}"
    downloaded_raster_request = httpx.get(url)
    return downloaded_raster_request.content


# Create date with zeroed out time and dummy UTC timezone at YYYY-MM-DD
def parse_date_string(date_str: str):
    return (
        datetime.fromisoformat(date_str)
        .astimezone(timezone.utc)
        .replace(hour=0, minute=0, second=0, microsecond=0)
    )


# Download image by spatialPredictionMaps record id and return path
def download_prediction_image_by_record_id(record_id: str) -> bytes:
    record = client.collection("spatialPredictionMaps").get_one(
        record_id, query_params={"requestKey": None}
    )
    return download_raster_image_bytes_from_record(record)


def download_first_prediction_image_by_date_range(
    lagoslakeid: int, start_date: str, end_date: str
):
    spatial_predictions = get_prediction_records_by_date_range(
        lagoslakeid, start_date, end_date
    )
    # First value is the earliest record within the range
    record = spatial_predictions[0]
    downloaded_raster = download_raster_image_bytes_from_record(record)
    return downloaded_raster, record.date


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
