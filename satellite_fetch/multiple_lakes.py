from functions import export_raster_main, open_gee_project
import pandas as pd
import sys
import ee


def run_multiple_lakes(
    out_dir, start_date_range, end_date_range, df_path, lagosid_path
):
    print(df_path, lagosid_path)
    # read csvs
    df = pd.read_csv(df_path)

    lagosid = pd.read_csv(lagosid_path)
    print("CSV imported")

    # select relevant columns from lagosid
    lagosid = lagosid[["lagoslakei"]]
    df = pd.concat([lagosid, df], axis=1)
    # by merging, we only have lakes with insitu data
    # To get ALL ids, just use lagosid csv

    df = df[df["chl_a"] < 2000]

    df = df.drop_duplicates(subset=["lagoslakei"])

    df = df[["lagoslakei", "site"]]
    df.to_csv("test.csv")

    successful_filenames = []
    for index, row in df.iterrows():
        lakeid = int(row["lagoslakei"])
        name = row["site"]
        formatted_name = name.lower().replace(" ", "-")
        print(f"Lakeid: {lakeid} Name: {formatted_name}")

        scale = 20
        # while not successful
        while scale <= 40:
            try:
                filename = f"{formatted_name}.tif"
                export_raster_main(
                    out_dir,
                    filename,
                    lakeid,
                    start_date_range,
                    end_date_range,
                    scale,
                )
                successful_filenames.append(filename)
                break  # If doesn't error, then just break
            except ee.ee_exception.EEException as error:
                if str(error).endswith("must be less than or equal to 50331648 bytes."):
                    scale += 10  # Raise the scale by 10 and try again
                elif str(error).endswith(
                    "Parameter 'object' is required."
                ):  # no images found
                    print("No images found")  # ignore
                    break
                else:
                    raise error
            except TypeError as error:
                if str(error).endswith("not recognized as a supported file format."):
                    # this happens if you write zero bytes to a tif with rasterio
                    # TypeError: 'out_all/black-lake.tif' not recognized as a supported file format.

                    scale += 10
                else:
                    raise error

    print(successful_filenames)
    print(f"{len(successful_filenames)} successful images found out of {len(df)} rows")
    # Get ONE image for a random date for every lake


if __name__ == "__main__":
    project = sys.argv[1]
    out_dir = sys.argv[2]
    start_date_range = sys.argv[3]  # STR, in format YYYY-MM-DD
    end_date_range = sys.argv[4]  # STR, in format YYYY-MM-DD
    tidy_df_path = sys.argv[5]
    lagosid_path = sys.argv[6]
    print(sys.argv)

    open_gee_project(project=project)

    run_multiple_lakes(
        out_dir,
        project,
        start_date_range,
        end_date_range,
        tidy_df_path,
        lagosid_path,
    )
