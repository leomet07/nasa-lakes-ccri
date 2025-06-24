from datetime import datetime as dt
from datetime import timedelta
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
import os

def histograms_for_lake_classes(low, med, high, out_folder, month, year):
    classes = {'low': low, 'med': med, 'high': high}
    
    # generate a histogram for each class
    for class_name, class_data in classes.items():
        all_differences = []

        for lake_df in class_data:
            differences = lake_df['Observed_Chla'] - lake_df['Predicted_Chla']
            all_differences.extend(differences)

        if all_differences:

            counts, bin_edges = np.histogram(all_differences, bins=30)
            std_dev_counts = np.std(counts)

            fig, ax = plt.subplots(figsize=(12, 6))

            ax.hist(all_differences, bins=30, label=f'{class_name.capitalize()} (Observed Chl-a - Predicted Chl-a)', color='blue')
            ax.set_xlabel('Difference (µg/L)', fontsize=14)
            ax.set_ylabel('Number of Occurrences', fontsize=14)
            ax.set_title(f'{class_name.capitalize()} Class: (Observed Chl-a) - (Predicted Chl-a) For All Observations in {month}-{year} (µg/L)')
            ax.legend()

            # add text box with standard deviation
            textstr = f"Standard Deviation: {std_dev_counts:.2f}"
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)

            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            
            # save the plot to the specified output folder
            output_path = os.path.join(out_folder, f'hist_diff_{class_name}_{month}_{year}.png')
            plt.savefig(output_path)
            plt.close()

def main_hist_function(csv_path, out_folder, month, year): 
    df = pd.read_csv(csv_path)

    # filter it to one month/year
    year = str(year)
    filtered_df = df[df['Date'].str.contains(f'-{month}-')] # assuming it's passed in as ex. '08'
    filtered_df = filtered_df[filtered_df['Date'].str.startswith(year)] # assuming it's passed in as ex. '2022'
    print(f"Filtered df to {year}-{month}")

    # define classes
    low = []
    med = []
    high = []

    # get the max predicted value for each lake and classify accordingly
    for lake in filtered_df['Site'].unique():
        print(f"Debugging - lake = {lake}")
        # Get the rows for the lake we're looking at
        lake_df = filtered_df.loc[filtered_df['Site'] == lake]
        print(f"Debugging - rows for this lake: {lake_df}")

        # get the predicted chl-a values for the lake we're looking at
        preds = lake_df['Predicted_Chla']
        max_val = np.max(preds)
        print(f"Debugging - max for this lake: {max_val}")

        if max_val < 10:
            low.append(lake_df)
            print(f"Debugging: Adding to low.")
        elif max_val < 20:
            med.append(lake_df)
            print(f"Debugging: Adding to med.")
        else:
            high.append(lake_df)
            print(f"Debugging: Adding to high.")

    histograms_for_lake_classes(low, med, high, out_folder, month, year)
    

if __name__ == "__main__":

    # Arguments:
    # csv_path: csv containing lake names, observed and predicted chlorophyll-a.
    # out_folder_name: folder name for lake histograms to be sent to.
    # month: month for which we want to see the difference in observed and predicted chlorophyll-a values.
    # year: year 

    if len(sys.argv) != 5:
        print("python gen_histograms_for_each_lake_classification.py csv_path out_folder_name month year")
        sys.exit(1)

    csv_path = sys.argv[1]
    out_folder = sys.argv[2]
    month = sys.argv[3]
    year = sys.argv[4]

    print("Debugging: about to run generate_histograms_all_lakes...")
    main_hist_function(csv_path, out_folder, month, year)

    # PIPELINE:
    # pass into our other function: 
    #   csv with in-situ/predictions
    #   get the max value for each lake for one month, year
    #   use that to classify the lake, i.e., add it to an array with the class name
    # THEN, for each class array:
    #   generate a histogram with a unique color for bars
