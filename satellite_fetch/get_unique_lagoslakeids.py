import os
# Setup dotenv
from dotenv import load_dotenv

load_dotenv()
import pandas as pd

nys_lagosid_path = os.getenv("nys_lagosid_path") 
nys_lagosid = pd.read_csv(nys_lagosid_path)
nys_lagosid = nys_lagosid[["lagoslakei"]]  # select relevant columns from lagosid
nys_lagosid = nys_lagosid.drop_duplicates(subset="lagoslakei")
nys_lagosid.to_csv("insitulakeids.csv", index=False)
print(nys_lagosid)
all_lagosid_path = os.getenv("all_lagosid_path")
all_lagosid = pd.read_csv(all_lagosid_path)
# all_lagosid = all_lagosid[["lagoslakei"]]  # select relevant columns from lagosid
all_lagosid = all_lagosid.drop_duplicates(subset="lagoslakeid")
all_lagosid.to_csv("all_lagos.csv", index=False)
print(all_lagosid.columns)

import geopandas as gp

gp.GeoDataFrame()

# LAGOS_NY_4ha_polygon's is Jillian's polygon version
# allNY_lakes_4ha is a "point version", useful for drawing points on a map to click on
shp = gp.GeoDataFrame.from_file(os.getenv("allNY_lakes_4ha_shp"))

print(shp.columns)

shp = shp[["GNIS_Name", "lagoslakei","geometry"]]
shp = shp.drop_duplicates(subset="lagoslakei")

shp.to_csv("all_nys_lakes_lagoslakeids.csv", index=False)

print(shp)


## THIS IS JUST A UTIL SCRIPT
## TO RUN ONE TIME BEFORE THE LAKES IDS ARE ADDED

from pocketbase import PocketBase  # Client also works the same

print(os.getenv("PUBLIC_POCKETBASE_URL"),os.getenv("POCKETBASE_ADMIN_EMAIL"), os.getenv("POCKETBASE_ADMIN_PASSWORD"))
client = PocketBase(os.getenv("PUBLIC_POCKETBASE_URL"))
admin_data = client.admins.auth_with_password(os.getenv("POCKETBASE_ADMIN_EMAIL"), os.getenv("POCKETBASE_ADMIN_PASSWORD"))


# Upload to DB
from tqdm import tqdm
from pprint import pprint

for index, row in tqdm(shp.iterrows()):
    name = row["GNIS_Name"]
    lagoslakeid = int(row["lagoslakei"])
    if name == "" or name == None:
        name = (
            f"lake{lagoslakeid}"  # In case lake name not found (doesn't really matter)
        )
    formatted_name = name.lower().replace(" ", "-")

    point_parsed = str(row["geometry"])[8:].replace("(","").replace(")","").split(" ")
    latitude = float(point_parsed[1])
    longitude = float(point_parsed[0])

    lake_body = {
        "name" : formatted_name,
        "lagoslakeid" : lagoslakeid,
        "latitude" : latitude,
        "longitude" :longitude
    }

    client.collection("lakes").create(lake_body)
