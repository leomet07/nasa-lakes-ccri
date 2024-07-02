#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages/libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import seaborn as sns
import geopandas as gpd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import time
import os

print("imports done")


# In[2]:


def prepare_data(df_path, lagosid_path, lulc_path, random_state=621, test_size=0.1):
    # read csvs
    df = pd.read_csv(df_path)
    lagosid = pd.read_csv(lagosid_path)
    print("CSV imported")

    # select relevant columns from lagosid
    lagosid = lagosid[['MEAN_lat', 'MEAN_long', 'lagoslakei']]
    df = pd.concat([lagosid, df], axis=1)

    # filter to Sentinel-2
    # df = df.loc[df['satellite'].isin(["1","2"])]

    # filter to LC
    # df = df.loc[df['satellite'].isin(["LC08","LC09"])]
    df = df[df['443'].notna()]

    # filter to low chl-a
    # df = df[df['chl_a'] < 15]

    # filter to strict date range
    # df = df[df['date_diff'] < 4]

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

    # left join df with iws_human
    df = df.merge(iws_human, on="lagoslakei")

    # set dfs
    # filter nas for morphology
    df = df[df['SA'].notna()]
    df = df[df['Max.depth'].notna()]

    # filter nas if using landuse
    df = df[df['pct_dev'].notna()]
    df = df[df['pct_ag'].notna()]

    features = df[['443', '493', '560', '665', '864', 'SA', 'Max.depth', 'pct_dev', 'pct_ag']]
    feature_names = ['443', '493', '560', '665', '864', 'SA', 'Max.depth', 'pct_dev', 'pct_ag']

    features = features.to_numpy()
    chla = df[["chl_a"]].to_numpy()
    weights = df[["weight"]].to_numpy()
    lat = df[["MEAN_lat"]].to_numpy()
    lon = df[["MEAN_long"]].to_numpy()
    coordinates = (df.MEAN_lat, df.MEAN_long)

    # squeeze converts dataframe to series
    lagosid = df[["lagoslakei"]].squeeze()

    # initialize random effects covariate
    Z = np.full((lagosid.size, 1), 1)

    # initial train-test split (holdout 10% for final training)
    x_train, x_test, y_train, y_test, wt_train, wt_test, lon_train, lon_test, lat_train, lat_test, clusters_train, clusters_test, z_train, z_test = train_test_split(
        features, chla, weights, lon, lat, lagosid, Z, test_size=test_size, random_state=random_state)

    return x_train, x_test, y_train, y_test, wt_train, wt_test, lon_train, lon_test, lat_train, lat_test, clusters_train, clusters_test, z_train, z_test



# define evaluate function
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def evaluate(model, test_features, test_labels, mod_name):
    predictions = model.predict(test_features)
    mean_dev = np.mean(predictions) - np.mean(test_labels)
    mae = mean_absolute_error(test_labels, predictions)
    pearson = stats.pearsonr(predictions, test_labels.ravel())[0]
    slope = np.sum(np.multiply((test_labels - test_labels.mean()),\
                            np.reshape(predictions - predictions.mean(),(-1,1))))\
        / np.sum(np.square((test_labels - test_labels.mean())))
    r2 = r2_score(test_labels, predictions)
    rmse = mean_squared_error(test_labels, predictions, squared=False)
    print('Model Performance')
    print('Mean Deviation: {:0.3f}'.format(np.mean(mean_dev)))
    print('Mean Absolute Error: {:0.3f}'.format(mae))
    print('RMSE: {:0.3f}'.format(rmse))
    print('Slope = {:0.3f}'.format(slope))
    print('Pearson correlation = {:0.3f}'.format(pearson))
    print('r2 = {:0.3f}'.format(r2))
    values = [mean_dev, mae, rmse, slope, pearson, r2]
    metrics = pd.DataFrame(values, 
                           columns = ['statistic'])
    metrics.index = ['MeanDeviation','MAE','RMSE', 'Slope',
               'Pearson', 'r2']
    # metrics.to_csv('model_outputs/model_statistics_' + mod_name +'.csv', index=True)
    return predictions


def train_model():

    # define paths to the data
    df_path = "ccri_tidy_chla_processed_data_V2.csv"
    lagosid_path = "ccri_lakes_withLagosID.csv"
    lulc_path = "LAGOSNE_iws_lulc105.csv"

    # call function
    x_train, x_test, y_train, y_test, wt_train, wt_test, lon_train, lon_test, lat_train, lat_test, clusters_train, clusters_test, z_train, z_test = prepare_data(df_path, lagosid_path, lulc_path)


    # In[3]:




    # In[4]:


    # get best params for rf

    rs = 621

    #define scoring metrics
    scoring = {"R2": "r2", 
            "MAE": "neg_mean_absolute_error", 
            "MAPE": "neg_mean_absolute_percentage_error"}


    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['log2', 'sqrt', 1.0]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [4, 7, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [4, 7, 10]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf}
    # pprint(random_grid)

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, 
                                param_distributions = random_grid, 
                                n_iter = 5, cv = 10, 
                                verbose=2, scoring = scoring,
                                    refit = "MAE",
                                random_state=rs, n_jobs = -1)

    time_start = time.time()

    print("Fitting...", time_start)

    # Fit the random search model
    rf_random.fit(x_train, y_train.ravel())

    time_end = time.time()
    time_diff = time_end - time_start
    print("Fitted at ", time_end)
    print(f"Elapsed time for fitting: {time_diff} seconds")

    #%%
    rf_random.best_params_
    # rf_random.cv_results_


    # In[5]:


    # check best params
    rs=621
    best_rf = RandomForestRegressor(random_state=rs,
                                    max_depth = 20,
                                    max_features = 1.0,
                                    min_samples_leaf = 7,
                                    min_samples_split = 10,
                                    n_estimators=1200,                                
                                    n_jobs=-1) 

    best_rf.fit(x_train, y_train.ravel(),sample_weight = wt_train.ravel())

    rf_evaluate = evaluate(best_rf, x_test, y_test, 'RF_best')


    # In[6]:



    #write obs vs pred to csv for manual stat calc
    best_rf_csv = pd.DataFrame(rf_evaluate, columns=['pred_chla'])
    y_test_csv = pd.DataFrame(y_test, columns=['chla'])
    best_rf_csv = pd.concat([y_test_csv.reset_index(drop=True), best_rf_csv], axis=1)

    # best_rf_csv.to_csv('rf_obs_vs_pred.csv', index = False)
    joblib.dump(best_rf, 'best_rf_model.joblib')

    return best_rf


# In[10]:


## Making the four extra bands data from constants
import rasterio
import xarray as xr
import numpy as np

# load raster data
# open the existing raster file for reading
input_tif = 'S2_Chautauqua_v2.tif'
with rasterio.open(input_tif) as src:
    raster_data = src.read()
    profile = src.profile  # Get the profile of the existing raster

# define constants for new bands
SA_constant = 43343
Max_depth_constant = 618
pct_dev_constant = 7.609999999999999
pct_ag_constant = 48.980000000000004

# create new bands
SA_band = np.full_like(raster_data[0], SA_constant, dtype=raster_data.dtype)
Max_depth_band = np.full_like(raster_data[0], Max_depth_constant, dtype=raster_data.dtype)
pct_dev_band = np.full_like(raster_data[0], pct_dev_constant, dtype=raster_data.dtype)
pct_ag_band = np.full_like(raster_data[0], pct_ag_constant, dtype=raster_data.dtype)

# update profile to reflect additional bands
profile.update(count=9)  # (update count to include original bands + 4 new bands)

# output GeoTIFF file
output_tif = 'S2_Chautauqua_v2_modified.tif'

# write the modified raster data to the new GeoTIFF file
with rasterio.open(output_tif, 'w', **profile) as dst:
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

print("Created a modified TIF with the four extra bands data from constants")


model_path = 'best_rf_model.joblib'

if os.path.isfile(model_path): # if the model has been trained already
    model = joblib.load(model_path)
else:
    model = train_model()


input_tif = 'S2_Chautauqua_v2_modified.tif'

with rasterio.open(input_tif) as src:
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

print("Predicting...", time_start)

# perform the prediction
predictions = model.predict(raster_data_2d)

time_end = time.time()
time_diff = time_end - time_start
print("Predicted at ", time_end)
print(f"Elapsed predicting time: {time_diff} seconds")

# reshape the predictions back to the original raster shape
predictions_raster = predictions.reshape(n_rows, n_cols)

# save the prediction result as a new raster file
output_tif = 'S2_Chautauqua_v2_predictions.tif'

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
plt.imshow(predictions_raster, cmap='viridis')
plt.colorbar()
plt.title('Predicted Values')
plt.show()

print(f"Predictions saved to {output_tif}")


# In[ ]:




