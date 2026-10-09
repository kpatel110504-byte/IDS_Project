"""02_preprocessing.py
Prepare data for models (very simple English comments).
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

MODELS_DIR = os.path.join("results","models")
os.makedirs(MODELS_DIR, exist_ok=True)

def load_datasets(train_path=os.path.join("data","KDDTrain.csv"),
                  test_path=os.path.join("data","KDDTest.csv")):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def get_label_column_name(df):
    return "label" if "label" in df.columns else df.columns[-1]

def make_binary_label(series):
    if series.dtype == object:
        return series.str.lower().str.contains("normal").apply(lambda x: 0 if x else 1).to_numpy()
    return series.apply(lambda x: 0 if x == 0 else 1).to_numpy()

def preprocess():
    train_df, test_df = load_datasets()
    label_col = get_label_column_name(train_df)
    print("Using label column:", label_col)

    y_train = make_binary_label(train_df[label_col])
    y_test = make_binary_label(test_df[label_col])

    X_train_raw = train_df.drop(columns=[label_col])
    X_test_raw = test_df.drop(columns=[label_col])

    combined = pd.concat([X_train_raw, X_test_raw], axis=0)
    combined_enc = pd.get_dummies(combined)

    n_train = len(X_train_raw)
    X_train_enc = combined_enc.iloc[:n_train,:]
    X_test_enc = combined_enc.iloc[n_train:,:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_test_scaled = scaler.transform(X_test_enc)

    X_train = X_train_scaled.astype("float32")
    X_test = X_test_scaled.astype("float32")
    y_train = y_train.astype("int64")
    y_test = y_test.astype("int64")

    np.save(os.path.join(MODELS_DIR,"X_train.npy"), X_train)
    np.save(os.path.join(MODELS_DIR,"X_test.npy"), X_test)
    np.save(os.path.join(MODELS_DIR,"y_train.npy"), y_train)
    np.save(os.path.join(MODELS_DIR,"y_test.npy"), y_test)

    print("Saved X_train, X_test, y_train, y_test in", MODELS_DIR)

if __name__ == "__main__":
    preprocess()
