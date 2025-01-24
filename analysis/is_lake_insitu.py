import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

insitu_lakes = pd.read_csv(os.getenv("INSITU_CSV_PATH"))

def is_lake_insitu(lagoslakeid: int):
    return (insitu_lakes["lagoslakei"] == lagoslakeid).any()

if __name__ == "__main__":
    print("47841: ", is_lake_insitu(47841))
    print("10536: ", is_lake_insitu(10536))
    print("49930: ", is_lake_insitu(49930))
    print("7241: ", is_lake_insitu(7241))
    print("6478: ", is_lake_insitu(6478))
    print("7153: ", is_lake_insitu(7153))