import db_utils
from pprint import pprint
from tqdm import tqdm

occurances_dict = {}

all_spatial_predictions_list = list(
    db_utils.spatial_predictions_collection.find({})
)  # query all then filter, faster than searching and querying on db level

for spatial_prediction in all_spatial_predictions_list:
    lagoslakeid = spatial_prediction["lagoslakeid"]
    if lagoslakeid not in occurances_dict:
        occurances_dict[lagoslakeid] = 1
    else:
        occurances_dict[lagoslakeid] += 1

occurances_as_list = list(occurances_dict.items())

occurances_as_list.sort(
    key=lambda x: x[
        1
    ],  # sort on the second part of tuple, aka the number of occurances
    reverse=True,
)

for pair in occurances_as_list[:10]:  # top ten occurances
    lagoslakeid, num_occurances = pair

    matched_lake = db_utils.lakes_collection.find_one({"lagoslakeid": lagoslakeid})

    print(
        f"{matched_lake["name"]} (lagoslakeid - {lagoslakeid}): {num_occurances} occurances"
    )
