import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import seaborn as sns

# This script generates box and whisker plots for the purpose of error analysis.

def main_box_plot_function_chla(csv_path, out_folder): 
    df = pd.read_csv(csv_path)
    # Categorize observed Chl-a
    df['Chla_Category'] = pd.cut(df['Observed_Chla'], bins=[-np.inf, 10, 20, np.inf], labels=['Low', 'Medium', 'High'])
    box_plot_chla(df, out_folder)
    
def box_plot_chla(filtered_df, out_folder):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Chla_Category', y='Predicted_Chla', data=filtered_df)
    yerr = np.abs(filtered_df['Observed_Chla'] - filtered_df['Predicted_Chla'])
    plt.errorbar(filtered_df['Chla_Category'], filtered_df['Predicted_Chla'], 
                 yerr=yerr, fmt='o', color='red', alpha=0.5)
    plt.xlabel('Observed Chl-a Category', fontsize=14)
    plt.ylabel('Predicted Chl-a (µg/L)', fontsize=14)
    plt.title('Predicted Chl-a vs Observed Chl-a Categories', fontsize=16)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Save the plot to the specified output folder
    output_path = os.path.join(out_folder, 'boxplot_predicted_vs_observed_categories.png')
    plt.savefig(output_path)
    plt.close()

def main_box_plot_function_depth(csv_path1, csv_path2, out_folder): 
    df = pd.read_csv(csv_path1)
    depth = pd.read_csv(csv_path2)

    # Merge the dataframes on the lake name
    depth.rename(columns={'site': 'Site'}, inplace=True)
    merged_df = pd.merge(df, depth[['Site', 'Max.depth']], on='Site', how='left')

    # Convert Max Depth from feet to meters and filter
    merged_df = merged_df[merged_df['Max.depth'] <= 700]
    merged_df['Max.depth'] = merged_df['Max.depth'] * 0.3048
    

    # Categorize Max Depth
    merged_df['Depth_Category'] = pd.qcut(merged_df['Max.depth'], q=3, labels=['Low', 'Medium', 'High'])

    box_plot_depth(merged_df, out_folder)

def box_plot_depth(merged_df, out_folder):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Depth_Category', y='Predicted_Chla', data=merged_df)
    yerr = np.abs(merged_df['Observed_Chla'] - merged_df['Predicted_Chla'])
    plt.errorbar(merged_df['Depth_Category'], merged_df['Predicted_Chla'], 
                 yerr=yerr, fmt='o', color='red', alpha=0.5)
    plt.xlabel('Max Depth Category', fontsize=14)
    plt.ylabel('Predicted Chl-a (µg/L)', fontsize=14)
    plt.title('Predicted Chl-a vs Max Depth Categories', fontsize=16)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Save the plot to the specified output folder
    output_path = os.path.join(out_folder, 'boxplot_predicted_vs_max_depth_categories.png')
    plt.savefig(output_path)
    plt.close()

def main_box_plot_function_sa(csv_path, csv_morphology, out_folder):
    df = pd.read_csv(csv_path)
    sa = pd.read_csv(csv_morphology)

    # Merge the dataframes on the lake name
    sa.rename(columns={'site':'Site'}, inplace=True)
    merged_df = pd.merge(df, sa[['Site', 'SA']], on='Site', how='left')

    # Convert Surface Area from acres to km²
    merged_df['SA_y'] = merged_df['SA_y'] * 0.00404686

    # Categorize Surface Area
    merged_df['SA_Category'] = pd.qcut(merged_df['SA_y'], q=3, labels=['Low', 'Medium', 'High'])

    box_plot_sa(merged_df, out_folder)

def box_plot_sa(merged_df, out_folder):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='SA_Category', y='Predicted_Chla', data=merged_df)
    plt.xlabel('Surface Area Category', fontsize=14)
    plt.ylabel('Predicted Chl-a (µg/L)', fontsize=14)
    plt.title('Predicted Chl-a vs Surface Area Categories', fontsize=16)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Save the plot to the specified output folder
    output_path = os.path.join(out_folder, 'boxplot_predicted_vs_surface_area_categories.png')
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python gen_box_plots.py csv_path_predictions csv_path_morphology out_folder_name variable")
        sys.exit(1)

    csv_path = sys.argv[1]
    morphology_csv = sys.argv[2]
    out_folder = sys.argv[3]
    var = sys.argv[4]

    print("Generating box plot...")
    if var == "chla":
        main_box_plot_function_chla(csv_path, out_folder)
    elif var == "depth":
        main_box_plot_function_depth(csv_path, morphology_csv, out_folder)
    elif var == "sa":
        main_box_plot_function_sa(csv_path, morphology_csv, out_folder)
