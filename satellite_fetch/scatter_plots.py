import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import seaborn as sns
from scipy.stats import gaussian_kde

# This script generates scatter plots for the purpose of error analysis.


# error vs. observed, all lakes, all dates
def main_scatter_function_chla(csv_path, out_folder): 
    df = pd.read_csv(csv_path)
    scatter_plot_chla(df, out_folder)
    
def scatter_plot_chla(filtered_df, out_folder):
    all_errors = []
    all_observed_chla = []

    for lake in filtered_df['Site'].unique():
        lake_df = filtered_df.loc[filtered_df['Site'] == lake]
        errors = lake_df['Observed_Chla'] - lake_df['Predicted_Chla']
        observed_chla = lake_df['Observed_Chla']
        all_errors.extend(errors)
        all_observed_chla.extend(observed_chla)

    if all_errors and all_observed_chla:
        # calculate point density
        xy = np.vstack([all_observed_chla, all_errors])
        z = gaussian_kde(xy)(xy)

        # sort the points by density
        idx = z.argsort()
        all_observed_chla = np.array(all_observed_chla)[idx]
        all_errors = np.array(all_errors)[idx]
        z = z[idx]

        fig, ax = plt.subplots(figsize=(12, 6))
        scatter = ax.scatter(all_observed_chla, all_errors, c=z, s=50, edgecolor='none', cmap='viridis')
        plt.colorbar(scatter, label='Density')
        ax.set_xlabel('Observed Chl-a (µg/L)', fontsize=14)
        ax.set_ylabel('Error (Observed - Predicted Chl-a) (µg/L)', fontsize=14)
        ax.set_title('Error vs Observed Chl-a', fontsize=16)

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        # save the plot to the specified output folder
        output_path = os.path.join(out_folder, 'scatter_error_vs_observed.png')
        plt.savefig(output_path)
        plt.close()



def main_scatter_function_depth(csv_path1, csv_path2, out_folder): 
    df = pd.read_csv(csv_path1)
    depth = pd.read_csv(csv_path2)

    # merge the dataframes on the lake name
    depth.rename(columns={'site': 'Site'}, inplace=True)
    merged_df = pd.merge(df, depth[['Site', 'Max.depth']], on='Site', how='left')

    scatter_plot_depth(merged_df, out_folder)

def scatter_plot_depth(merged_df, out_folder):
    all_errors = []
    all_max_depth = []

    for lake in merged_df['Site'].unique():
        lake_df = merged_df.loc[merged_df['Site'] == lake]
        errors = lake_df['Observed_Chla'] - lake_df['Predicted_Chla']
        max_depth = lake_df['Max.depth']
        
        # remove NaNs / infinite values
        mask = ~np.isnan(errors) & ~np.isnan(max_depth) & ~np.isinf(errors) & ~np.isinf(max_depth)
        errors = errors[mask]
        max_depth = max_depth[mask]
        
        all_errors.extend(errors)
        all_max_depth.extend(max_depth)

    if all_errors and all_max_depth:
        # calculate point density
        xy = np.vstack([all_max_depth, all_errors])
        z = gaussian_kde(xy)(xy)

        # sort the points by density so the densest points are plotted last
        idx = z.argsort()
        all_max_depth = np.array(all_max_depth)[idx]
        all_errors = np.array(all_errors)[idx]
        z = z[idx]

        fig, ax = plt.subplots(figsize=(12, 6))
        scatter = ax.scatter(all_max_depth, all_errors, c=z, s=50, edgecolor='none', cmap='viridis')
        plt.colorbar(scatter, label='Density')
        ax.set_xlabel('Max Depth (m)', fontsize=14)
        ax.set_ylabel('Error (Observed - Predicted Chl-a) (µg/L)', fontsize=14)
        ax.set_title('Error vs Max Depth', fontsize=16)

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        # save the plot to the specified output folder
        output_path = os.path.join(out_folder, 'scatter_error_vs_max_depth.png')
        plt.savefig(output_path)
        plt.close()

def main_scatter_function_sa(csv_path, csv_morphology, out_folder):
    df = pd.read_csv(csv_path)
    sa = pd.read_csv(csv_morphology)

    # merge
    sa.rename(columns={'site':'Site'}, inplace=True)
    print(sa['SA'])
    merged_df = pd.merge(df, sa[['Site', 'SA']], on='Site', how='left')
    print("Columns in merged_df:", merged_df.columns)
    print("First few rows of merged_df:\n", merged_df.head())

    scatter_plot_sa(merged_df, out_folder)

def scatter_plot_sa(merged_df, out_folder):
    all_errors = []
    all_sa = []

    for lake in merged_df['Site'].unique():
        lake_df = merged_df.loc[merged_df['Site'] == lake]
        errors = lake_df['Observed_Chla'] - lake_df['Predicted_Chla']
        sa = lake_df['SA_y']


        mask = ~np.isnan(errors) & ~np.isnan(sa) & ~np.isinf(errors) & ~np.isinf(sa)
        errors = errors[mask]
        sa = sa[mask]
        
        all_errors.extend(errors)
        all_sa.extend(sa)

    if all_errors and all_sa:
        xy = np.vstack([all_sa, all_errors])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        all_sa = np.array(all_sa)[idx]
        all_errors=np.array(all_errors)[idx]
        z = z[idx]

        fig, ax = plt.subplots(figsize=(12,6))
        scatter = ax.scatter(all_sa, all_errors, c=z, s=50, edgecolor ='none', cmap='viridis')
        plt.colorbar(scatter, label='Density')
        ax.set_xlabel('Surface Area (m^2)', fontsize=14)
        ax.set_ylabel('Error (Observed - Predicted Chl-a) (µg/L)', fontsize=14)
        ax.set_title('Error vs Surface Area', fontsize=16)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    output_path = os.path.join(out_folder, 'scatter_error_vs_surface_area.png')
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python gen_scatter_plots.py csv_path_predictions csv_path_morphology out_folder_name variable")
        sys.exit(1)

    csv_path = sys.argv[1]
    morphology_csv = sys.argv[2]
    out_folder = sys.argv[3]
    var = sys.argv[4]

# /Users/erinfoley/Desktop/nasa2024/data/ccri_tidy_chla_processed_data_V2.csv
    print("Generating scatter plot...")
    if var == "chla":
        main_scatter_function_chla(csv_path, out_folder)
    elif var == "depth":
        main_scatter_function_depth(csv_path, morphology_csv, out_folder)
    elif var == "sa":
        main_scatter_function_sa(csv_path, morphology_csv, out_folder)
