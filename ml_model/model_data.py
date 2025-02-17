# Setup dotenv
from dotenv import load_dotenv
import os
import pandas as pd
import time
import numpy as np

load_dotenv()

NAN_SUBSTITUTE_CONSANT = -99999

# Creating the proper dataframe (csv) from other datasets
training_df_path = os.getenv("INSITU_CHLA_TRAINING_DATA_PATH") # training data
lagosid_path = os.getenv("CCRI_LAKES_WITH_LAGOSID_PATH") # needed to get lagoslakeid for training data entries
lulc_path = os.getenv("LAGOS_LAKE_INFO_PATH") # General lake info (for constants)


def prepare_data(df_path, lagosid_path, lulc_path):
    # read csvs
    df = pd.read_csv(df_path)
    lagosid = pd.read_csv(lagosid_path)

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

    SA_SQ_KM_DF = pd.read_csv("lagosID_area.csv")
    df = df.merge(SA_SQ_KM_DF, on="lagoslakei")
    print(df)
    return df, iws_human # Array of pandas dataframes


def prepared_cleaned_data(unclean_data): # Returns CUDF df
    unclean_data = unclean_data[['chl_a', '443', '493', '560', '665','703', '740', '780', '834', '864', 'SA_SQ_KM_FROM_SHAPEFILE','pct_dev','pct_ag']]
    unclean_data = unclean_data.fillna(NAN_SUBSTITUTE_CONSANT)
    input_cols = ['443', '493', '560', '665','703', '740', '780', '834', '864', 'SA_SQ_KM_FROM_SHAPEFILE','pct_dev','pct_ag']
    for col in unclean_data.select_dtypes(["object"]).columns:
        unclean_data[col] = unclean_data[col].astype("category").cat.codes.astype(np.int32)

    # cast all columns to int32
    for col in unclean_data.columns:
        unclean_data[col] = unclean_data[col].astype(np.float32)  # needed for random forest

    # put target/label column first [ classic XGBoost standard ]
    output_cols = ["chl_a"] + input_cols
    # unclean_data.to_csv("preindexaspandas.csv")
    unclean_data = unclean_data.reindex(columns=output_cols)

    return unclean_data # Now it is clean

# define constants for new bands
def get_constants(lakeid):
    filtered_df = all_data[all_data['lagoslakei'] == lakeid] # maybe this lake is from insitu and we know it's depth and surface area?
    
    # SA = filtered_df['SA'].iloc[0] if not filtered_df.empty else NAN_SUBSTITUTE_CONSANT # NAN_SUBSTITUTE_CONSANT = null in our cudf random forest
    # Max_depth = filtered_df['Max.depth'].iloc[0]  if not filtered_df.empty else NAN_SUBSTITUTE_CONSANT
    SA_SQ_KM = filtered_df['SA_SQ_KM_FROM_SHAPEFILE'].iloc[0]
    pct_dev = lagos_lookup_table['pct_dev'].iloc[0] # Lagos look up table should have this
    pct_ag = lagos_lookup_table['pct_ag'].iloc[0] # Lagos look up table should have this

    return SA_SQ_KM, pct_dev, pct_ag


all_data, lagos_lookup_table = prepare_data(training_df_path, lagosid_path, lulc_path) # Returns insitu points merged with lagoslookup table AND lagoslookup table for all non-insitu lakes as well
cleaned_data = prepared_cleaned_data(all_data)
cleaned_data = cleaned_data[cleaned_data['chl_a'] < 350] # most values are 0-100, remove the crazy 4,000 outlier
print(cleaned_data)