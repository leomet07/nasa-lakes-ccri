import sys
import pandas as pd
import os

def gen_pred_hab_map(csv_path, out_folder, year):
    # read the csv
    year = str(year)
    print(f"Debugging: converted year to string: {year}")
    print("Debugging: about to read csv.")
    df = pd.read_csv(csv_path)

    # get the unique lakes
    unique_lakes = df.drop_duplicates("Site").reset_index(drop=True)
    lat = unique_lakes['MEAN_lat']
    lon = unique_lakes['MEAN_long']
    print(f"Size of lat: {len(lat)}")
    print(f"Size of lon: {len(lon)}")
    print(f"Debugging: number of unique lakes: {len(unique_lakes)}")

    df_output = pd.DataFrame({
        'Lake': unique_lakes['Site'],
        'Lat': lat,
        'Lon': lon,
        'HAB Detected:': [False] * len(unique_lakes)
    })

    dont_have_data = []

    print("Debugging: df_output generation successful.")
    print(df_output)

    # get the original dataframe (i.e., all observations) just for this year
    filtered_df = df[df['Date'].str.startswith(year)]
    print(f"Debugging... filtered df to {year}")

    # for each lake, get the data from the specified year
    for i in range(len(unique_lakes['Site'])):
        lake = unique_lakes['Site'][i]
        print(f"Debugging... lake: {lake}")

        # filter the data for the current lake
        lake_data = filtered_df[filtered_df['Site'] == lake]

        if not lake_data.empty:
            print(f"Debugging... example of observation for this lake in {year}: ", lake_data.iloc[0])
             # then, if its max predicted chlorophyll-a value is above 20,
            # mark it in the output dataframe.
            print(f"DEBUGGING HERE RN: ", lake_data['Predicted_Chla'])
            if lake_data['Predicted_Chla'].max() > 20:
                print(f"HAB Detected for {lake} in {year}!")
                df_output.loc[i, 'HAB Detected:'] = True
        else:
            print(f"Debugging... No lake data for this year.")
            dont_have_data.append(lake)

    # make new folder if necessary
    if not os.path.exists(out_folder): 
        os.makedirs(out_folder) 
    
    # save the output dataframe as a csv
    file_path = os.path.join(out_folder, f"NYS_HAB_Predictions_{year}.csv")
    df_output.to_csv(file_path, index=False)
    print(f"{len(dont_have_data)} lakes don't have in-situ data for {year}.")
    print("Debugging: CSV saved successfully.")


if __name__ == "__main__":

    # Arguments: 
    # csv_path: the path to the csv with predictions/observations. MATCHES IN-SITU
    # out_folder_name: the folder where you want to save the image to.
    # year: the program decides whether there is a predicted HAB in each lake in the specified year.

    # Output: The ultimate output of this script is an image of New York State
    # where each point represents a lake in New York State. If the point is white,
    # the maximum PREDICTED chlorophyll-a value is < 20 (i.e., no HAB detected).
    # If the point is green, the maximum PREDICTED chlorophyll-a value is > 20 (i.e.,
    # HAB detected). We are using our in-situ/prediction csv to generate this map.

    # It will also print out the number of predicted HAB lakes and the percentage
    # of predicted HAB lakes.

    if len(sys.argv) != 4:
        print("python gen_time_series.py <csv_path> <out_folder_name> <year>")
        sys.exit(1)

    csv_path = sys.argv[1]
    out_folder = sys.argv[2]
    year = sys.argv[3]

    print(f"Debugging: csv_path={csv_path}")
    print(f"Debugging: out_folder={out_folder}")
    print(f"Debugging: year specified={year}")
    
    gen_pred_hab_map(csv_path, out_folder, year)