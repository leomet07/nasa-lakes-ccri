import pandas as pd
import geopandas as gpd
import folium
import webbrowser
import os
import sys

def gen_pred_hab_map(csv_path, out_folder):
    df = pd.read_csv(csv_path)
    print("Debugging: read csv.")

    map_ny = folium.Map(location=[42.8282, -75.5447], zoom_start=7)

    shape_map = gpd.read_file('/Users/erinfoley/Desktop/NYS_Civil_Boundaries.shp/State.shp')
    folium.GeoJson(data=shape_map).add_to(map_ny)

    for i, row in df.iterrows():
        if row['1020']:
            color = '#c9ffd3'
        elif row['20']:
            color = '#00aa20'
        else:
            color ='blue'
        folium.CircleMarker(
            location=[row['Lat'], row['Lon']],
            radius=5,
            fill=True,
            color='black',
            weight=1,
            fill_color=color,
            fill_opacity=1,
        ).add_to(map_ny)
        print("Added dot!")

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    map_file = os.path.join(out_folder, 'NYS_HAB_Map.html')
    map_ny.save(map_file)
    print(f"Map saved as {map_file}.")

    abs_path = os.path.abspath(map_file)
    webbrowser.open('file://' + abs_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("python gen_histograms.py csv_path out_folder_name")
        sys.exit(1)

    csv_path = sys.argv[1]
    out_folder = sys.argv[2]

    print(f"Debugging: generating map of predicted HABs.")
    gen_pred_hab_map(csv_path, out_folder)


