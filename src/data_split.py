import pandas as pd
from sklearn.model_selection import train_test_split


def split_data():
    df = pd.read_csv("data/raw/adult.csv")

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["income"]
    )

    train_df.to_csv("data/raw/train.csv", index=False)
    test_df.to_csv("data/raw/test.csv", index=False)
