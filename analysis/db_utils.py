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
    record = client.collection("spatialPredictionMaps").get_one(record_id)
    return download_raster_image_bytes_from_record(record)


def download_first_prediction_image_by_date_range(
    lagoslakeid: int, start_date: str, end_date: str
):
    start_date_obj = parse_date_string(start_date)
    end_date_obj = parse_date_string(end_date)

    # This is probably best done database side, need to add lagoslikeid field to prediction collection
    # Because if there are 1000+ spatialPredictions, this will be a lot of network requests
    lake = client.collection("lakes").get_first_list_item(f"lagoslakeid={lagoslakeid}")
    spatial_predictions = lake.spatial_predictions

    spatial_predictions_expanded = []
    for spatial_prediction_id in spatial_predictions:
        spatial_prediction = client.collection("spatialPredictionMaps").get_one(
            spatial_prediction_id
        )
        spatial_prediction.date = datetime.fromisoformat(spatial_prediction.date)
        if (
            spatial_prediction.date > end_date_obj
            or spatial_prediction.date < start_date_obj
        ):
            continue  # Date is outside of range

        spatial_predictions_expanded.append(spatial_prediction)

    spatial_predictions_expanded.sort(key=(lambda a: a.date))

    # First value is the earliest record within the range
    record = spatial_predictions_expanded[0]
    downloaded_raster = download_raster_image_bytes_from_record(record)
    return downloaded_raster, record.date
