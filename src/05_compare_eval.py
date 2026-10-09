"""05_compare_eval.py
Compare RF, SVM and PyTorch NN.
"""
import os
import numpy as np
import joblib
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pytorch_nn import SimpleNN

MODELS_DIR = os.path.join("results","models")

def load_test():
    X_test = np.load(os.path.join(MODELS_DIR,"X_test.npy"))
    y_test = np.load(os.path.join(MODELS_DIR,"y_test.npy"))
    return X_test, y_test

def eval_classical(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    return dict(model=name, accuracy=acc, precision=prec, recall=rec, f1=f1)

def eval_nn(X_test, y_test):
    input_dim = X_test.shape[1]
    model = SimpleNN(input_dim)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR,"pytorch_nn_model.pth"), map_location="cpu"))
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
        probs = torch.sigmoid(logits).numpy()
        y_pred = (probs >= 0.5).astype(int).flatten()
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    return dict(model="PyTorch NN", accuracy=acc, precision=prec, recall=rec, f1=f1)

def main():
    X_test, y_test = load_test()
    rf = joblib.load(os.path.join(MODELS_DIR,"random_forest_model.joblib"))
    svm = joblib.load(os.path.join(MODELS_DIR,"svm_model.joblib"))

    results = [
        eval_classical("Random Forest", rf, X_test, y_test),
        eval_classical("SVM", svm, X_test, y_test),
        eval_nn(X_test, y_test),
    ]

    print("=== MODEL COMPARISON (TEST) ===")
    print("{:<15} {:>9} {:>9} {:>9} {:>9}".format("Model","Acc","Prec","Recall","F1"))
    for r in results:
        print("{:<15} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f}".format(
            r["model"], r["accuracy"], r["precision"], r["recall"], r["f1"]
        ))

    with open(os.path.join(MODELS_DIR,"model_comparison.txt"),"w") as f:
        f.write("Model,Accuracy,Precision,Recall,F1\n")
        for r in results:
            f.write(f"{r['model']},{r['accuracy']:.4f},{r['precision']:.4f},{r['recall']:.4f},{r['f1']:.4f}\n")

if __name__ == "__main__":
    main()
