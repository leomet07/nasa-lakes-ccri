from datetime import datetime as dt
from datetime import timedelta
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
import os
import statistics


# Generate Histogram of difference between observed and predictions, all lakes, one month
def hist_diff_all_lakes_one_month(df, out_folder):
    new_df = pd.DataFrame({
        'Observed':df['Observed_Chla'],
        'Predicted':df['Predicted_Chla'],
        'Difference':df['Observed_Chla']-df['Predicted_Chla']
    })

    counts, bin_edges = np.histogram(new_df['Difference'], bins=30)
    std_dev_counts = np.std(counts)

    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(new_df['Difference'], bins=30, label='Observed Chl-a - Predicted Chl-a', color='blue')
    ax.set_xlabel('Difference (µg/L)', fontsize=14)
    ax.set_ylabel('Number of Occurrences', fontsize=14)
    ax.set_title(
        f'(Observed Chl-a) - (Predicted Chl-a) For All Observations in {month}-{year} (µg/L)'
    )
    ax.legend()
    
    # add text box with standard deviation
    textstr = f"Standard Deviation: {std_dev_counts:.2f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    file_path = os.path.join(out_folder, f"hist_diff_all_lakes_{year}{month}.png")
    plt.savefig(file_path)
    plt.close()
    
    print("Saved and closed plot.")

def main_hist_diff(csv_path, out_folder, month, year):
    df = pd.read_csv(csv_path)
    print("Read csv.")

    # Filter to one month
    year = year + ""
    filtered_df = df[df['Date'].str.contains(f'-{month}-')] # assuming it's passed in as ex. '08'
    filtered_df = filtered_df[filtered_df['Date'].str.startswith(year)] # assuming it's passed in as ex. '2022'
    print(f"Filtered df to {year}-{month}")

    hist_diff_all_lakes_one_month(filtered_df, out_folder)


# Generate Histogram of difference between observed and predictions, all lakes that have max value < 10



# Generate Histogram of difference between observed and predictions, all lakes that have 10 value < max < 20

# Generate Histogram of difference between observed and predictions, all lakes that have max value >= 20


# Add figures, add discussion


# Function to generate a histogram of observed vs. predicted chlorophyll-a concentrations
# for a given lake
def generate_histogram(df, out_folder):
    columns = ["Site", "Observed_Chla", "Predicted_Chla", "Date"]
    df = df[columns]
    print("Debugging: about to generate plot...")
    plt.figure(figsize=(12, 6))
    plt.hist(df["Observed_Chla"], bins=30, alpha=0.5, label="Observed", color="blue")
    plt.hist(df["Predicted_Chla"], bins=30, alpha=0.3, label="Predicted", color="red")
    plt.xlabel("Chlorophyll-a Concentration (µg/L)", fontsize=14)
    plt.ylabel("Number of Instances", fontsize=14)
    plt.ylim(0, 500)
    plt.title(
        "Observed vs Predicted Chlorophyll-a Concentrations for All Lakes", fontsize=16
    )
    plt.legend(loc="upper right", fontsize=12)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    file_path = os.path.join(out_folder, "hist_all_lakes_v?.png")
    plt.savefig(file_path)
    plt.close()
    print("Debugging: saved combined plot, closed plot.")


def main_hist(csv_path, out_folder):
    print("Debugging: about to read csv...")
    df = pd.read_csv(csv_path)
    print(f"Number of obs: {len(df)}")
    print("Debugging: about to generate histogram...")
    generate_histogram(df, out_folder)
    print("Saved combined histogram succesfully.")


if __name__ == "__main__":

    # Arguments:
    # csv_path: csv containing lake names, observed and predicted chlorophyll-a.
    # out_folder_name: folder name for lake histograms to be sent to.
    # month: month for which we want to see the difference in observed and predicted chlorophyll-a values.
    # year: year 

    if len(sys.argv) != 5:
        print("python gen_histograms.py csv_path out_folder_name month year")
        sys.exit(1)

    csv_path = sys.argv[1]
    out_folder = sys.argv[2]
    month = sys.argv[3]
    year = sys.argv[4]

    print("Debugging: about to run generate_hisograms_all_lakes...")
    # main_hist(csv_path, out_folder)
    main_hist_diff(csv_path, out_folder, month, year)
