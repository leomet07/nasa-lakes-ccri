import db_utils
from pprint import pprint

lakes = db_utils.client.collection("lakes").get_full_list(batch=100_000)
lakes.sort(key=lambda x: len(x.spatial_predictions))

for lake in lakes:
    print(
        f"{lake.name}, lagoslakeid: {lake.lagoslakeid}, # of predictions: {len(lake.spatial_predictions)}"
    )
