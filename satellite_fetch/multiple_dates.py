from functions import export_raster_main, open_gee_project
import datetime
import pandas as pd
import sys

if __name__ == "__main__":
    if len(sys.argv) != 9:
        print(
            "python multiple_dates.py <out_dir> <project> <lakeid> <start_date> <end_date> <scale> <frequency> <out_filename>"
        )
        sys.exit(1)

    project = sys.argv[1]
    out_dir = sys.argv[2]
    lakeid = int(sys.argv[3])
    start_date_range = sys.argv[4]  # STR, in format YYYY-MM-DD
    end_date_range = sys.argv[5]  # STR, in format YYYY-MM-DD
    scale = int(sys.argv[6])
    frequency = int(sys.argv[7])  # in days
    out_filename_template = sys.argv[8]

    date_range = list(
        pd.date_range(start=start_date_range, end=end_date_range, freq=f"{frequency}D")
    )

    open_gee_project(project=project)

    successful_filenames = []

    for i in range(len(date_range) - 1):  # stop iterating one element short
        start_date = date_range[i].strftime(f"%Y-%m-%d")
        end_date = date_range[i + 1].strftime(f"%Y-%m-%d")
        print(f"Start date: {start_date}, End date: {end_date}")

        out_filename = out_filename_template + start_date + "to" + end_date + ".tif"

        try:
            export_raster_main(
                out_dir, out_filename, project, lakeid, start_date, end_date, scale
            )
            successful_filenames.append(out_filename)
        except Exception as e:
            print(e)

    print(successful_filenames)
    print(
        f"{len(successful_filenames)} successful images found out of {len(date_range)} subintervals"
    )
