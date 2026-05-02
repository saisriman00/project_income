import pandas as pd
import os

URL = "https://raw.githubusercontent.com/pooja2512/Adult-Census-Income/master/adult.csv"


def ingest_data():
    os.makedirs("data/raw", exist_ok=True)
    df = pd.read_csv(URL)
    df.to_csv("data/raw/adult.csv", index=False)
    return df
