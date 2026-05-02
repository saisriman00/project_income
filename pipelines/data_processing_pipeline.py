import pandas as pd
import numpy as np
import joblib
import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


def process_data():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    train_df = pd.read_csv("data/raw/train.csv")
    test_df = pd.read_csv("data/raw/test.csv")

    # clean strings
    for col in train_df.select_dtypes(include="object").columns:
        train_df[col] = train_df[col].str.strip()
        test_df[col] = test_df[col].str.strip()

    # replace missing
    train_df.replace(["?", " ?"], np.nan, inplace=True)
    test_df.replace(["?", " ?"], np.nan, inplace=True)

    target = "income"

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # remove duplicates
    mask = ~X_train.duplicated()
    X_train = X_train[mask]
    y_train = y_train[mask]

    # encode target
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # columns
    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols)
    ])

    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    # save
    joblib.dump(preprocessor, "artifacts/preprocessor.pkl")
    joblib.dump(le, "artifacts/label_encoder.pkl")
    joblib.dump(X_train.columns.tolist(), "artifacts/feature_columns.pkl")

    pd.DataFrame(X_train_prep).to_csv(
        "data/processed/X_train.csv", index=False)
    pd.DataFrame(X_test_prep).to_csv("data/processed/X_test.csv", index=False)
    pd.DataFrame(y_train_enc).to_csv("data/processed/y_train.csv", index=False)
    pd.DataFrame(y_test_enc).to_csv("data/processed/y_test.csv", index=False)
