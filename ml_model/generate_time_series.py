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
from sklearn.model_selection import train_test_split

if IS_CPU_MODE:
    import cpu_model_training as model_training
else:
    import model_training

mode = "all"
out_path = os.path.join("timeseries", "timeseries_" + mode)
if not os.path.exists(out_path):
    os.makedirs(out_path)

andrew_model = model_training.andrew_model

all_insitu_predicted = andrew_model.predict(model_training.X.values)

results_df = pd.DataFrame(
    columns=[
        "lagoslakei",
        "date",
        "site",
        "observed_chla",
        "predicted_chla",
    ]
)

all_data = model_data.all_data_cleaned
all_predictions = all_insitu_predicted

# Use split all data cleaned
if mode.lower() == "testing":
    all_data_train, all_data_test = train_test_split(
        all_data, test_size=0.2, random_state=621
    )  # constant hard coded from ./ml_model/model_data.py
    all_data = all_data_test
    all_predictions = model_training.y_pred

print(
    f"All data len: {len(all_data)} | All insitu predicted: {len(all_predictions)}<- Should be the same"
)  # should be the same


for index in tqdm(range(len(all_data))):
    row = all_data.iloc[index]

    results_df.loc[len(results_df)] = [
        row["lagoslakei"],
        row["sample_date"],  # actual sample data of the water
        row["site"],  # lake name
        row["chl_a"],  # observed
        all_predictions[index],  # predicted
    ]

results_df["date"] = pd.to_datetime(results_df["date"])

# Credit to Erin Foley for orignally making this time series plot function
for lakeid in tqdm(np.unique(all_data["lagoslakei"])):
    df = results_df[results_df["lagoslakei"] == lakeid]  # matching row to this lakeid
    lake_name = df.iloc[0]["site"]  # once lakeid filtered, just get lake name

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
    file_path = os.path.join(out_path, f"{lake_name}_time_series.jpg")

    plt.savefig(file_path, dpi=400)
    plt.close()
