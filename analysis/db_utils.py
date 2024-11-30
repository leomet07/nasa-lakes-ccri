import os
from dotenv import load_dotenv
from pocketbase import PocketBase  # Client also works the same
import httpx
import rasterio

load_dotenv()

client = PocketBase(os.getenv("PUBLIC_POCKETBASE_URL"))
admin_data = client.admins.auth_with_password(
    os.getenv("POCKETBASE_ADMIN_EMAIL"), os.getenv("POCKETBASE_ADMIN_PASSWORD")
)


# Download image by spatialPredictionMaps record id and return path
def download_prediction_image_by_record_id(id: str) -> bytes:
    record = client.collection("spatialPredictionMaps").get_one(id)
    url = f"https://db.nyhabmonitor.site/api/files/{record.collection_id}/{record.id}/{record.raster_image}"
    downloaded_raster_request = httpx.get(url)
    downloaded_raster = downloaded_raster_request.content
    return downloaded_raster


# TODO:
# Implement a download first prediciton image by lagoslakeid within in a certian date range
