# preprocessing.py

import pandas as pd
import os


def preprocess_data():
    df = pd.read_csv(
        "C:\\Users\\anush\\OneDrive\\Desktop\\HealthWise-Project\\data\\heart.csv"
    )
    df.dropna(inplace=True)

    # Use one-hot encoding to match app.py
    df = pd.get_dummies(df, drop_first=True)
    return df
