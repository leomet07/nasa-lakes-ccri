import sys
import pandas as pd
import os

def gen_pred_hab_map(csv_path, out_folder, year, lagos):

    # read the csv
    print("Debugging: about to read csv.")
    df = pd.read_csv(csv_path) # Lake ID and Max
    lagos = pd.read_csv(lagos) # Lake ID and lat lon
    print(df.head)

    lagos = lagos.drop_duplicates('site') 
    # merge dataframes so that we have ID, coordinates, max predicted value
    with_coords = pd.merge(lagos, df, on='lagoslakei')

    df_output = pd.DataFrame({
        'Lake': with_coords['lagoslakei'], # have this already
        'Lat': with_coords['MEAN_lat'],
        'Lon': with_coords['MEAN_long'],
        '1020': [False] * len(with_coords), # for all lakes we have an image for, add a val for prediction
        '20': [False] * len(with_coords)
    })

    print("Debugging: df_output generation successful.")
    print(with_coords)

    # for each lake, get the max prediction
    for i in range(len(with_coords)):
        lake = with_coords.iloc[i]
        print(f"Debugging... lake: {lake}")

        if lake['max'] > 10 and lake['max'] < 20:
            df_output.loc[i, '1020'] = True
        elif lake['max'] >= 20:
            df_output.loc[i, '20'] = True


    # make new folder if necessary
    if not os.path.exists(out_folder): 
        os.makedirs(out_folder) 
    
    # save the output dataframe as a csv
    file_path = os.path.join(out_folder, f"NYS_HAB_Predictions_{year}.csv")
    df_output.to_csv(file_path, index=False)
    print("Debugging: CSV saved successfully.")


if __name__ == "__main__":

    # path for lagos: /Users/erinfoley/Desktop/nasa2024/data/ccri_lakes_withLagosID.csv
    # path for max val csv:
    # year considering rn: 2022

    # Arguments: 
    # csv_path: the path to the csv that has site, coords, max_prediction from the prediction raster of a given date's chla vals
    # out_folder_name: the folder where you want to save the image to.

    # Output: The ultimate output of this script is an image of New York State
    # where each point represents a lake in New York State. If the point is white,
    # the maximum PREDICTED chlorophyll-a value is < 20 (i.e., no HAB detected).
    # If the point is green, the maximum PREDICTED chlorophyll-a value is > 20 (i.e.,
    # HAB detected). We are using our *prediction images* to generate this map.

    # It will also print out the number of predicted HAB lakes and the percentage
    # of predicted HAB lakes.

    if len(sys.argv) != 5:
        print("python gen_time_series.py <csv_path> <out_folder_name> <year> <lagos_csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    out_folder = sys.argv[2]
    year = sys.argv[3]
    lagos = sys.argv[4]

    
    gen_pred_hab_map(csv_path, out_folder, year, lagos)