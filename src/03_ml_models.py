"""03_ml_models.py
Train Random Forest and SVM with easy comments.
"""
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

MODELS_DIR = os.path.join("results","models")
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data():
    X_train = np.load(os.path.join(MODELS_DIR,"X_train.npy"))
    X_test = np.load(os.path.join(MODELS_DIR,"X_test.npy"))
    y_train = np.load(os.path.join(MODELS_DIR,"y_train.npy"))
    y_test = np.load(os.path.join(MODELS_DIR,"y_test.npy"))
    return X_train, X_test, y_train, y_test

def eval_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    print(f"=== {name} ===")
    print("Accuracy :", f"{acc:.4f}")
    print("Precision:", f"{prec:.4f}")
    print("Recall   :", f"{rec:.4f}")
    print("F1-score :", f"{f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print()
    return dict(model=name, accuracy=acc, precision=prec, recall=rec, f1=f1)

def main():
    X_train, X_test, y_train, y_test = load_data()

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    print("Training Random Forest...")
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(MODELS_DIR,"random_forest_model.joblib"))
    rf_metrics = eval_model("Random Forest", rf, X_test, y_test)

    svm = SVC(kernel="rbf", probability=True, random_state=42)
    print("Training SVM...")
    svm.fit(X_train, y_train)
    joblib.dump(svm, os.path.join(MODELS_DIR,"svm_model.joblib"))
    svm_metrics = eval_model("SVM", svm, X_test, y_test)

    with open(os.path.join(MODELS_DIR,"ml_metrics_summary.txt"),"w") as f:
        for m in (rf_metrics, svm_metrics):
            f.write(f"Model: {m['model']}\n")
            f.write(f"  Accuracy : {m['accuracy']:.4f}\n")
            f.write(f"  Precision: {m['precision']:.4f}\n")
            f.write(f"  Recall   : {m['recall']:.4f}\n")
            f.write(f"  F1-score : {m['f1']:.4f}\n\n")

if __name__ == "__main__":
    main()
