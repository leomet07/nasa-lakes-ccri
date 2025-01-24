# Setup dotenv
from dotenv import load_dotenv

load_dotenv()

from pocketbase import PocketBase  # Client also works the same
from pocketbase.client import FileUpload
import os

client = PocketBase(os.getenv("PUBLIC_POCKETBASE_URL"))
admin_data = client.admins.auth_with_password(os.getenv("POCKETBASE_ADMIN_EMAIL"), os.getenv("POCKETBASE_ADMIN_PASSWORD"))


def add_suffix_to_filename(filename : str, suffix : str):
  parts = filename.split(".")
  return parts[0] + f"_{suffix}." + ".".join(parts[1:])


import cudf
import pandas as pd

def prepare_data(df_path, lagosid_path, lulc_path, random_state=621, test_size=0.1):
    # read csvs
    df = pd.read_csv(df_path)

    lagosid = pd.read_csv(lagosid_path)
    print("CSV imported")

    # select relevant columns from lagosid
    lagosid = lagosid[['lagoslakei']]
    df = pd.concat([lagosid, df], axis=1)

    df = df[df['chl_a'] < 2000]

    # create chl-a mathematical inputs
    df["NDCI"] = ((df['703'] - df['665']) / (df['703'] + df['665']))
    df["NDVI"] = ((df['864'] - df['665']) / (df['864'] + df['665']))
    df["3BDA"] = (((df['493'] - df['665']) / (df['493'] + df['665'])) - df['560'])

    # create rough vol based on depth & SA
    df["lake_vol"] = (df['SA'] * df['Max.depth'])

    # load lagos-ne iws data, merged with dataframe
    iws_lulc = pd.read_csv(lulc_path)

    # summarize percent developed land
    iws_lulc['iws_nlcd2011_pct_dev'] = iws_lulc['iws_nlcd2011_pct_21'] + \
        iws_lulc['iws_nlcd2011_pct_22'] + iws_lulc['iws_nlcd2011_pct_23'] + \
        iws_lulc['iws_nlcd2011_pct_24']

    # summarize percent agricultural land
    iws_lulc['iws_nlcd2011_pct_ag'] = iws_lulc['iws_nlcd2011_pct_81'] + iws_lulc['iws_nlcd2011_pct_82']

    # create new dataframe with the two variables
    iws_human = iws_lulc[['lagoslakeid', 'iws_nlcd2011_pct_dev', 'iws_nlcd2011_pct_ag']]

    iws_human = iws_human.rename(columns={'lagoslakeid': 'lagoslakei', 'iws_nlcd2011_pct_dev': 'pct_dev', 'iws_nlcd2011_pct_ag': 'pct_ag'})
    # At this point, iws_human contains 50k rows (ALL lagos lakes, including the 4400 NYS lakes)
    # print("one of 4k lakes after: ", iws_human.loc[lambda x: x['lagoslakei'] == 57171])
    # print("# of rows in iws_human: ", len(iws_human))

    # left join df with iws_human, which cuts # of lakes in df to 360 (14k entries of in situ readings)
    df = df.merge(iws_human, on="lagoslakei")
    return df, iws_human # Array of pandas dataframes


def prepared_cleaned_data(unclean_data): # Returns CUDF df
    unclean_data = unclean_data[['chl_a', '443', '493', '560', '665','703', '740', '780', '834', '864','SA','Max.depth','pct_dev','pct_ag']]
    unclean_data = unclean_data.fillna(-99999)
    input_cols = ['443', '493', '560', '665','703', '740', '780', '834', '864','SA','Max.depth','pct_dev','pct_ag']
    for col in unclean_data.select_dtypes(["object"]).columns:
        unclean_data[col] = unclean_data[col].astype("category").cat.codes.astype(np.int32)

    # cast all columns to int32
    for col in unclean_data.columns:
        unclean_data[col] = unclean_data[col].astype(np.float32)  # needed for random forest

    # put target/label column first [ classic XGBoost standard ]
    output_cols = ["chl_a"] + input_cols
    unclean_data.to_csv("preindexaspandas.csv")
    unclean_data = unclean_data.reindex(columns=output_cols)

    unclean_data.to_csv("postindexaspandas.csv")
    df_cudf = cudf.from_pandas(unclean_data)
    return df_cudf



import pandas as pd
from cuml.ensemble import RandomForestRegressor
from dask_ml.model_selection import RandomizedSearchCV
from cuml.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
import cudf
import time
from pathlib import Path
import os

# Load the CSV file OR create it
# Creating the proper dataframe (csv) from other datasets
df_path = "ccri_tidy_chla_processed_data_V2.csv"
lagosid_path = "ccri_lakes_withLagosID.csv"
lulc_path = "LAGOSNE_iws_lulc105.csv"
all_data, lagos_lookup_table = prepare_data(df_path, lagosid_path, lulc_path) # Returns insitu points merged with lagoslookup table AND lagoslookup table for all non-insitu lakes as well
cleaned_data = prepared_cleaned_data(all_data)
cleaned_data = cleaned_data[cleaned_data['chl_a'] < 350] # most values are 0-100, remove the crazy 4,000 outlier

# Loading the csv of an already existing dataframe
# file_path = '/content/drive/MyDrive/NASA/lenny_google_collab/postindexaspandas.csv'
# data = cudf.read_csv(file_path)
# data = data.drop(columns=['Unnamed: 0'])
# cleaned_data = data[data['chl_a'] < 350] # most values are 0-100, remove the crazy 4,000 outlier


# Convert float columns to integers
for col in cleaned_data.columns:
    cleaned_data[col] = cleaned_data[col].astype(np.float32)

X = cleaned_data.drop(columns=['chl_a'])
y = cleaned_data['chl_a']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=621)
print("Dataframes created and data split successfully.")


# define constants for new bands

def get_constants(df_all, lakeid):
    filtered_df = df_all[df_all['lagoslakei'] == lakeid] # maybe this lake is from insitu and we know it's depth and surface area?
    
    SA = filtered_df['SA'].iloc[0] if not filtered_df.empty else -99999 # -99999  null in our cudf random forest
    Max_depth = filtered_df['Max.depth'].iloc[0]  if not filtered_df.empty else -99999
    pct_dev = lagos_lookup_table['pct_dev'].iloc[0] # Lagos look up table should have this
    pct_ag = lagos_lookup_table['pct_ag'].iloc[0] # Lagos look up table should have this

    return SA, Max_depth, pct_dev, pct_ag


# In[7]:


import joblib
import rasterio
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd


def modify_tif(input_tif : str, SA_constant : float, Max_depth_constant : float, pct_dev_constant: float, pct_ag_constant : float) -> str:
    with rasterio.open(input_tif) as src:
        raster_data = src.read()
        profile = src.profile  # Get the profile of the existing raster


    # create new bands
    SA_band = np.full_like(raster_data[0], SA_constant, dtype=raster_data.dtype)
    Max_depth_band = np.full_like(raster_data[0], Max_depth_constant, dtype=raster_data.dtype)
    pct_dev_band = np.full_like(raster_data[0], pct_dev_constant, dtype=raster_data.dtype)
    pct_ag_band = np.full_like(raster_data[0], pct_ag_constant, dtype=raster_data.dtype)

    # update profile to reflect additional bands
    profile.update(count=13)  # (update count to include original bands + 4 new bands)

    # output GeoTIFF file
    modified_tif = add_suffix_to_filename(input_tif, "modified")

    # write the modified raster data to the new GeoTIFF file
    with rasterio.open(modified_tif, 'w', **profile) as dst:
        # write original bands
        for i in range(1, raster_data.shape[0] + 1):
            dst.write(raster_data[i-1], indexes=i)

        # write additional bands
        dst.write(SA_band, indexes=raster_data.shape[0] + 1)
        dst.write(Max_depth_band, indexes=raster_data.shape[0] + 2)
        dst.write(pct_dev_band, indexes=raster_data.shape[0] + 3)
        dst.write(pct_ag_band, indexes=raster_data.shape[0] + 4)


        dst.transform = src.transform
        dst.crs = src.crs

    # print(f"Created {modified_tif} with the four extra bands data from constants")
    return modified_tif


# In[8]:


'''
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features =  ['log2', 'sqrt', 1.0] # all in the following array are fast but log2 is fastest ['log2', 'sqrt', 1.0]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [4,7,10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [4,7,10]

# Define the parameter grid for RandomizedSearchCV
param_grid = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'max_features': max_features,
    'split_criterion': [2] # 2 is MSE (source: https://stackoverflow.com/questions/77984286/cuml-randomforestclassifier-typeerror-an-integer-is-required)
}


# Initialize the Random Forest model
rf_model = RandomForestRegressor() #2 for mse

time_start = time.time()
print("Random search starting...")


# Initialize the RandomizedSearchCV with 5-fold cross-validation
random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_grid,
    n_iter=5,
    scoring='neg_mean_absolute_error', # neg_mean_absolute_error = MAE
    cv=10, # cv of 10 has the same accuracy but runs slower
    #verbose=2,
    # random_state=42,
    n_jobs=-1
)

# Fit the RandomizedSearchCV to find the best hyperparameters
random_search.fit(X_train.to_numpy(), y_train.to_numpy())
time_end = time.time()
time_diff = time_end - time_start
print(f"Random search finished, elapsed {time_diff} seconds")

# Get the best parameters
best_params = random_search.best_params_
best_model = random_search.best_estimator_

time_start = time.time()
print("Predicting...")
# Predict on the test set using the best model
y_pred = best_model.predict(X_test)
time_end = time.time()
time_diff = time_end - time_start
print(f"Predicted! Elapsed {time_diff} seconds")

# Calculate the Mean Squared Error and r2
r2 = r2_score(y_test.to_numpy(), y_pred.to_numpy())
rmse = mean_squared_error(y_test.to_numpy(), y_pred.to_numpy()) ** 0.5
print(f"Best Parameters: {best_params}")
print(f"r2 score: {r2}")
print(f"RMSE: {rmse}")
# print(f"First 10 Predictions: {y_pred[:10]}")
'''


# In[9]:


good_known_params =  {
                      'n_estimators': 200, 'min_samples_split': 4, 'min_samples_leaf': 4, 'max_features': 'log2', 'max_depth': 30}

andrew_params = {
     'max_depth': 30, # Andrew params
    'max_features': 'sqrt',
    'min_samples_leaf' :1,
    'min_samples_split' :2,
    'n_estimators':1200,
}

# ^ not an interval

# known_fast_model = RandomForestRegressor(**good_known_params)
andrew_model = RandomForestRegressor(**andrew_params)

time_start = time.time()
print("Known fit starting...")

# Fit the RandomizedSearchCV to find the best hyperparameters
andrew_model.fit(X_train.to_numpy(), y_train.to_numpy())
time_end = time.time()
time_diff = time_end - time_start
print(f"Known fit finished, elapsed {time_diff} seconds")

time_start = time.time()
print("Predicting...")
# Predict on the test set using the best model
y_pred = andrew_model.predict(X_test)
time_end = time.time()
time_diff = time_end - time_start
print(f"Predicted! Elapsed {time_diff} seconds")

# Calculate the Mean Squared Error and r2
r2 = r2_score(y_test.to_numpy(), y_pred.to_numpy())
rmse = mean_squared_error(y_test.to_numpy(), y_pred.to_numpy()) ** 0.5
print(f"r2 score: {r2}")
print(f"RMSE: {rmse}")



import matplotlib.pyplot as plt
import rasterio


def predict(input_tif : str, id: int, display = True):
    modified_tif = add_suffix_to_filename(input_tif, "modified")
    with rasterio.open(modified_tif) as src:
        raster_data = src.read()
        profile = src.profile
        transform = src.transform
        crs = src.crs

    # reshape raster data to 2D array for model prediction
    n_bands, n_rows, n_cols = raster_data.shape
    raster_data_2d = raster_data.reshape(n_bands, -1).T

    # handle NaN values by replacing them with the mean of each band
    nan_mask = np.isnan(raster_data_2d)
    means = np.nanmean(raster_data_2d, axis=0)
    raster_data_2d[nan_mask] = np.take(means, np.where(nan_mask)[1])


    time_start = time.time()

    #print(f"Predicting on {modified_tif}")
    #print("Time start: ", time_start)

    # perform the prediction
    # NOTE::::: best_model (from a search upon interval) or known_fast_model (from GIVEN params)
    # predictions = best_model.predict(raster_data_2d)
    # predictions = known_fast_model.predict(raster_data_2d)
    predictions = andrew_model.predict(raster_data_2d)

    time_end = time.time()
    time_diff = time_end - time_start
    #print("Predicted at ", time_end)
    #print(f"Elapsed predicting time: {time_diff} seconds")

    # max_value = predictions.max()
    # max_values[id]= max_value

    # reshape the predictions back to the original raster shape
    predictions_raster = predictions.reshape(n_rows, n_cols)

    # save the prediction result as a new raster file
    output_tif = add_suffix_to_filename(input_tif, "predicted")

    with rasterio.open(output_tif, 'w',
                      driver='GTiff',
                      height=n_rows,
                      width=n_cols,
                      count=1,
                      dtype=predictions_raster.dtype,
                      crs=crs,
                      transform=transform) as dst:
        dst.write(predictions_raster, 1)

    # plot the result
    if display:
        plt.imshow(predictions_raster, cmap='viridis')
        plt.colorbar()
        plt.title('Predicted Values')
        plt.show()

    #print(f"Predictions saved to {output_tif}")
    return output_tif, predictions_raster




def save_png(input_tif, out_folder, predictions_raster, date, scale, display=True):
    first_pixel_value = predictions_raster[0, 0]
    masked_raster = np.where(predictions_raster == first_pixel_value, np.nan, predictions_raster)

    min_value = 0
    max_value = 60
    increment = 5
    values = np.arange(min_value, max_value + increment, increment)

    fig = plt.figure(figsize=(10, 8))
    plt.imshow(masked_raster, cmap='viridis', interpolation='none', vmin=min_value, vmax=max_value)
    plt.axis('off')
    stem = Path(input_tif).stem

    # cbar = plt.colorbar()
    # cbar.set_label(f'Predicted chlorophyll-A in ug/L \n ({date}, scale: {scale})')
    # cbar.set_ticks(values)
    # cbar.set_ticklabels([str(val) for val in values])

    # png filename
    output_tif = add_suffix_to_filename(input_tif, "predicted")
    output_png = stem + ".png"
    output_png_path = os.path.join(out_folder, output_png)
    #print("Out folder: ", out_folder)
    #print("PNG name: ", output_png)
    #print(f"Saving png to {output_png_path}")
    # save the png
    plt.savefig(output_png_path, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    if display:
        plt.show()
    else:
        plt.close(fig)
    return output_png_path



from datetime import datetime
import json
import rasterio
from pyproj import Proj
import uuid
session_uuid = str(uuid.uuid4())


def upload_spatial_map(lakeid : int, raster_image_path: str, display_image_path : str, datestr : str, corners : list, scale : int):
    # Step 1: Find Lakeid
    filter_str = f"lagoslakeid='{lakeid}'"

    matched_lake = client.collection("lakes").get_list(1, 20, {"filter": filter_str}).items[0]
    lake_db_id = matched_lake.id

    # print("Found lake's id in the DB: ", lake_db_id)

    dateiso = datetime.strptime(datestr, '%Y-%m-%d').isoformat()
    # Step 2

    body = {
        "raster_image" : None,
        "display_image" : None,
        "date" : dateiso, # utc
        "corner1latitude": corners[0][0],
        "corner1longitude": corners[0][1],
        "corner2latitude": corners[1][0],
        "corner2longitude": corners[1][1],
        "scale" : 30,
        "session_uuid" : session_uuid,
        "lake" : lake_db_id,
        "lagoslakeid" : lakeid
    }

    # print("Inside upload:" , body)
    with open(raster_image_path, "rb") as rasterimage:
        body["raster_image"] = FileUpload((f"raster_image_{lakeid}_{datestr}.tif", rasterimage))
        with open(display_image_path, "rb") as displayimage:
            body["display_image"] = FileUpload((f"display_image_{lakeid}_{datestr}.png", displayimage))
            created = client.collection("spatialPredictionMaps").create(body)

    lake_result = client.collection("lakes").update(lake_db_id, {"spatialPredictions+" : created.id}) # here the library does not snakeCase

    # print("Lake result: ",lake_result.__dict__)



input_tif_folder = "five_augusts_4k_lakes" # Specify the folder inside of the tar
paths = os.listdir(input_tif_folder)

print("Number of files to run: ", len(paths))


import rasterio
from tqdm import tqdm

png_out_folder = "content/out_pngs_5summer_5klakes/"

if not os.path.exists(png_out_folder):
  os.makedirs(png_out_folder)

error_paths = []

print("Current session id: ", session_uuid)

import gc

for path_tif in tqdm(paths):
    path_tif = os.path.join(input_tif_folder, path_tif)
    try:
        with rasterio.open(path_tif) as raster:
            tags = raster.tags()
            id = int(tags["id"])
            date = tags["date"] # date does NOT do anything here, just for title
            scale = tags["scale"] # scale does NOT do anything here, just for title

            top_left = raster.transform * (0, 0)
            bottom_right = raster.transform * (raster.width, raster.height)
            crs = raster.crs

        p = Proj(crs)
        # Output is in the format: (lat, long)
        corner1 = list(p(top_left[0], top_left[1], inverse=True)[::-1])
        corner2 = list(p(bottom_right[0], bottom_right[1], inverse=True)[::-1])
        corners = [corner1, corner2]
        # print("id: ", id, " date: ", date, " scale: ", scale, " corners: ", corners)

        # Get constants
        SA_constant, Max_depth_constant, pct_dev_constant, pct_ag_constant = get_constants(all_data, id)
        # print("Constants based on id: ", SA_constant, Max_depth_constant, pct_dev_constant, pct_ag_constant)


        modified_path_tif = modify_tif(path_tif, SA_constant, Max_depth_constant, pct_dev_constant, pct_ag_constant)

        output_tif, predictions_loop = predict(path_tif, id, display = False)

        output_path_png = save_png(path_tif, png_out_folder, predictions_loop, date, scale, display = False)

        upload_spatial_map(id, output_tif, output_path_png, date, corners, scale)

        with open(f"{session_uuid}_status.txt", "a") as file_obj:
            file_obj.write(path_tif +"\n")

        gc.collect() # Clear memory after prediction
    except Exception as e:
        print("Error: ", e)
        error_paths.append(path_tif)

print(f"Successfully finished {len(paths)} uploads with {len(error_paths)} errors")
print("Session ID: ", session_uuid)

import json
with open(f"error_paths_{session_uuid}.json", "w") as file:
    file.write(json.dumps(error_paths))
