"""01_eda.py
Simple EDA script with very easy English comments.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_RESULTS_DIR = os.path.join("results", "eda")
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

def load_datasets(train_path=os.path.join("data","KDDTrain.csv"),
                  test_path=os.path.join("data","KDDTest.csv")):
    """Load train and test CSV files."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def get_label_column_name(df):
    """Return label column name, prefer 'label', else last column."""
    return "label" if "label" in df.columns else df.columns[-1]

def basic_info(train_df, test_df, label_col):
    print("Train shape:", train_df.shape)
    print("Test shape :", test_df.shape)
    print("\nTrain numeric summary:")
    print(train_df.describe())
    print("\nTrain label distribution (counts):")
    print(train_df[label_col].value_counts())
    print("\nTrain label distribution (percentage):")
    print((train_df[label_col].value_counts(normalize=True)*100).round(2))

    plt.figure()
    train_df[label_col].value_counts().plot(kind="bar")
    plt.title("Train: Normal vs Attack")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_RESULTS_DIR,"train_label_counts.png"))
    plt.close()

    plt.figure()
    test_df[label_col].value_counts().plot(kind="bar")
    plt.title("Test: Normal vs Attack")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_RESULTS_DIR,"test_label_counts.png"))
    plt.close()

def correlation_heatmap(train_df):
    numeric_df = train_df.select_dtypes(include=["int64","float64","int32","float32"])
    if numeric_df.shape[1] == 0:
        print("No numeric columns for correlation heatmap.")
        return
    corr = numeric_df.corr()
    plt.figure(figsize=(10,8))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation Heatmap (Train numeric features)")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_RESULTS_DIR,"correlation_heatmap.png"))
    plt.close()

def attack_type_distribution(train_df, label_col):
    perc = train_df[label_col].value_counts(normalize=True)*100
    print("\nAttack type percentage (train):")
    print(perc.round(2))
    plt.figure()
    perc.plot(kind="bar")
    plt.title("Attack Type Percentage (Train)")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_RESULTS_DIR,"attack_type_percentage_train.png"))
    plt.close()

def main():
    print("Loading datasets...")
    train_df, test_df = load_datasets()
    label_col = get_label_column_name(train_df)
    print("Using label column:", label_col)
    basic_info(train_df, test_df, label_col)
    correlation_heatmap(train_df)
    attack_type_distribution(train_df, label_col)
    print("EDA finished. Plots saved in", BASE_RESULTS_DIR)

if __name__ == "__main__":
    main()
