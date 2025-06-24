import requests
from datetime import datetime as dt
from datetime import timedelta
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pocketbase import PocketBase  # Client also works the same
from pocketbase.client import FileUpload
import json
import rasterio
from pyproj import Proj
from dotenv import load_dotenv

load_dotenv()
client = PocketBase(os.getenv("PUBLIC_POCKETBASE_URL"))
admin_data = client.admins.auth_with_password(os.getenv("POCKETBASE_ADMIN_EMAIL"), os.getenv("POCKETBASE_ADMIN_PASSWORD"))

def upload_spatial_map(lakeid : int, timeseries_path : str):
    # Step 1: Find Lakeid
    filter = f"lagoslakeid='{lakeid}'"
    matched_lake = client.collection("lakes").get_list(1, 20, {"filter": filter}).items[0]
    lake_db_id = matched_lake.id
    print("Found lake's id in the DB: ", lake_db_id)

    # Step 2
    body = {}
    with open(timeseries_path, "rb") as timeseries_:
        body["graph"] = FileUpload((f"timeseries_{lakeid}.png", timeseries_))
        created = client.collection("timeSeries").create(body)

    lake_result = client.collection("lakes").update(lake_db_id, {"timeSeries+" : created.id}) # here the library does not snakeCase
    print(lake_result.__dict__)

# Params:
#   lake = String - name of lake you want to generate a time series for
#   df = DataFrame - DataFrame containing predictions, observed values, date
#   out_path = String - path to save the time series images
def plot_observed_vs_predicted(df, lake, out_path):
    df.head()
    # get stuff we care about... i.e., specific lake, observations, predictions, and the date
    columns = ['Site', 'Observed_Chla', 'Predicted_Chla', 'Date']
    df = df[columns]
    df = df.loc[df['Site'].isin([lake])]
    df['Date'] = pd.to_datetime(df['Date'])

    # plot params
    plt.figure(figsize=(14, 8)) 
    plt.scatter(df['Date'], df['Predicted_Chla'], marker='x', label='Predicted', s=100)
    plt.scatter(df['Date'], df['Observed_Chla'], marker='o', label='Observed', s=100)

    # label the axes
    plt.xlabel('Date (Month-Year)', fontsize=18, weight='bold')  
    plt.ylabel('Chl-a: Predicted vs Observed (µg/L)', fontsize=18, weight='bold')  
    plt.title(f'Observed vs Predicted Chlorophyll-a Concentrations for {lake.title()}', fontsize=20, weight='bold')  

    # format the x-axis to show month and year from earliest in-situ observations (2013)
    x_axis = plt.gca().xaxis
    x_axis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    x_axis.set_major_locator(mdates.MonthLocator(interval=5))

    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="upper right", fontsize=16)
    
    if not os.path.exists(out_path): 
        os.makedirs(out_path) 

    lake_name = lake.title().replace(' ', '')
    file_path = os.path.join(out_path, f"{lake_name}_time_series.png") 
    
    plt.savefig(file_path)
    plt.close()

    return file_path

def generate_time_series_all_lakes(csv_path, lake_id_path, out_path):
    # read the csv
    df = pd.read_csv(csv_path)
    lake_ids = pd.read_csv(lake_id_path)

    # get the unique lake names
    unique_lakes = df['Site'].unique()
    print(f"Number of unique lakes: ", len(unique_lakes))

    # for every single lake we have, generate a time series
    for lake in unique_lakes:
        image_path = plot_observed_vs_predicted(df, lake, out_path)
        lake_id_row = lake_ids.loc[lake_ids['site'] == lake, 'lagoslakei']
        
        if not lake_id_row.empty:
            lake_id = lake_id_row.values[0]
            upload_spatial_map(lakeid=lake_id, timeseries_path=image_path)
        else:
            print(f"Lake ID not found for {lake}")

if __name__ == "__main__":

    # Arguments: 
    # csv_path: the path to the csv with predictions/observations.
    # lake_id_path: the path to the csv with lake IDs.
    # out_folder_name: the folder where you want to save this time series to.
    # api_url: the base URL of the PocketBase API.
    # collection: the name of the PocketBase collection to store data.

    if len(sys.argv) != 4:
        print(
            "python gen_time_series.py <csv_path> <lake_id_path> <out_folder_name> <api_url> <collection>"
        )
        sys.exit(1)

    csv_path = sys.argv[1]
    lake_id_path = sys.argv[2]
    out_folder = sys.argv[3]

    generate_time_series_all_lakes(
        csv_path,
        lake_id_path,
        out_folder
    )

    print("Data inserted into PocketBase successfully.")
