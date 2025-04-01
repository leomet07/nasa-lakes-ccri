# Setup dotenv
from dotenv import load_dotenv
import os

load_dotenv()

IS_CPU_MODE = os.getenv("IS_CPU_MODE").lower() == "true"

import model_data
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

if IS_CPU_MODE:
    import cpu_model_training as model_training
else:
    import model_training

out_path = "timeseries"
if not os.path.exists(out_path):
    os.makedirs(out_path)


andrew_model = model_training.andrew_model

all_insitu_predicted = andrew_model.predict(model_training.X.values)

print(
    f"All data len: {len(model_data.all_data_cleaned)} | Model training X len: {len(model_training.X)} | All insitu predicted: {len(all_insitu_predicted)}<- Should be the same"
)  # should be the same


results_df = pd.DataFrame(columns=["lagoslakei", "date", "site", "observed_chla", "predicted_chla",])
for index in tqdm(range(len(model_data.all_data_cleaned))):
    row = model_data.all_data_cleaned.iloc[index]

    results_df.loc[len(results_df)] = [
        row["lagoslakei"],
        row["sample_date"], # actual sample data of the water
        row["site"], # lake name
        row["chl_a"], # observed
        all_insitu_predicted[index], # predicted
    ]

results_df["date"] = pd.to_datetime(results_df["date"])

# Credit to Erin Foley for orignally making this time series plot function
for lakeid in tqdm(np.unique(model_data.all_data_cleaned["lagoslakei"])):
    df = results_df[results_df["lagoslakei"] == lakeid] # matching row to this lakeid
    lake_name = df.iloc[0]["site"] # once lakeid filtered, just get lake name

    # plot params
    plt.figure(figsize=(14, 8))
    plt.scatter(df["date"], df["predicted_chla"], marker="x", label="Predicted", s=100)
    plt.scatter(df["date"], df["observed_chla"], marker="o", label="Observed", s=100)

    # label the axes
    plt.xlabel("Date (Month-Year)", fontsize=18, weight="bold")
    plt.ylabel("Chl-a: Predicted vs Observed (µg/L)", fontsize=18, weight="bold")
    plt.title(
        f"Observed vs Predicted Chlorophyll-a Concentrations for {lake_name}",
        fontsize=20,
        weight="bold",
    )

    # format the x-axis to show month and year from earliest in-situ observations (2013)
    x_axis = plt.gca().xaxis
    x_axis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
    x_axis.set_major_locator(mdates.MonthLocator(interval=5))

    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="upper right", fontsize=16)

    lake_name = lake_name.title().replace(" ", "")
    file_path = os.path.join(out_path, f"{lake_name}_time_series.png")

    plt.savefig(file_path)
    plt.close()

