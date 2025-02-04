# Setup dotenv
from dotenv import load_dotenv
import os

load_dotenv()

import joblib
import cudf
from cuml.ensemble import RandomForestRegressor
from dask_ml.model_selection import RandomizedSearchCV
from cuml.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd
import time
import numpy as np

NAN_SUBSTITUTE_CONSANT = -99999

# Creating the proper dataframe (csv) from other datasets
training_df_path = os.getenv("INSITU_CHLA_TRAINING_DATA_PATH") # training data
lagosid_path = os.getenv("CCRI_LAKES_WITH_LAGOSID_PATH") # needed to get lagoslakeid for training data entries
lulc_path = os.getenv("LAGOS_LAKE_INFO_PATH") # General lake info (for constants)
DO_HYPERPARAM_SEARCH = os.getenv("DO_HYPERPARAM_SEARCH").lower() == "true"

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
    unclean_data = unclean_data.fillna(NAN_SUBSTITUTE_CONSANT)
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

all_data, lagos_lookup_table = prepare_data(training_df_path, lagosid_path, lulc_path) # Returns insitu points merged with lagoslookup table AND lagoslookup table for all non-insitu lakes as well
cleaned_data = prepared_cleaned_data(all_data)
cleaned_data = cleaned_data[cleaned_data['chl_a'] < 350] # most values are 0-100, remove the crazy 4,000 outlier

# Convert float columns to integers
for col in cleaned_data.columns:
    cleaned_data[col] = cleaned_data[col].astype(np.float32)

X = cleaned_data.drop(columns=['chl_a'])
y = cleaned_data['chl_a']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=621)
print("Dataframes created and data split successfully.")

# define constants for new bands
def get_constants(lakeid):
    filtered_df = all_data[all_data['lagoslakei'] == lakeid] # maybe this lake is from insitu and we know it's depth and surface area?
    
    SA = filtered_df['SA'].iloc[0] if not filtered_df.empty else NAN_SUBSTITUTE_CONSANT # NAN_SUBSTITUTE_CONSANT = null in our cudf random forest
    Max_depth = filtered_df['Max.depth'].iloc[0]  if not filtered_df.empty else NAN_SUBSTITUTE_CONSANT
    pct_dev = lagos_lookup_table['pct_dev'].iloc[0] # Lagos look up table should have this
    pct_ag = lagos_lookup_table['pct_ag'].iloc[0] # Lagos look up table should have this

    return SA, Max_depth, pct_dev, pct_ag

def hyper_param_search_and_train_model():
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
    print("Searching for best hyperparamaters using a random search and checking by fitting model...")

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

    # Fit the RandomizedSearchCV to find the best hyperparameters (by randomly trying combos within param grid and fitting and testing accordingly)
    random_search.fit(X_train.to_numpy(), y_train.to_numpy())
    time_end = time.time()
    time_diff = time_end - time_start
    print(f"Random search & Fit finished, elapsed {time_diff} seconds")

    # Get the best parameters
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_
    return best_params, best_model

if DO_HYPERPARAM_SEARCH:
    best_params, best_model = hyper_param_search_and_train_model()
    time_start = time.time()
    print("Predicting on test dataset with hyperparam optimized model...")
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

good_known_params =  {'n_estimators': 200, 'min_samples_split': 4, 'min_samples_leaf': 4, 'max_features': 'log2', 'max_depth': 30}

andrew_params = {
     'max_depth': 30, # Andrew params
    'max_features': 'sqrt',
    'min_samples_leaf' :1,
    'min_samples_split' :2,
    'n_estimators':1200,
}

print("Known fit starting...")
andrew_model = RandomForestRegressor(**andrew_params) # Instead of searching for params, use preconfigured params andrew found
time_start = time.time()
andrew_model.fit(X_train.to_numpy(), y_train.to_numpy())  # Fit model (which uses preconfigured params)
time_end = time.time()
time_diff = time_end - time_start
print(f"Known fit finished, elapsed {time_diff} seconds")

time_start = time.time()
print("Predicting on test...")
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
