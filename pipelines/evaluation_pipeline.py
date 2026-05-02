import pandas as pd
import joblib
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_pipeline():
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    results = []

    for file in os.listdir("artifacts/models"):
        model = joblib.load(f"artifacts/models/{file}")
        name = file.replace(".pkl", "")

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob)
        })

    df = pd.DataFrame(results).sort_values("F1", ascending=False)

    os.makedirs("artifacts", exist_ok=True)
    df.to_csv("artifacts/model_results.csv", index=False)

    best_model = df.iloc[0]["Model"]

    with open("artifacts/best_model.txt", "w") as f:
        f.write(best_model)

    return best_model
