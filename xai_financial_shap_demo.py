# xai_financial_shap_demo.py
# ----------------------------------------------------------------------------
# Self‐contained script to demo SHAP explanations for a financial classifier
# ----------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

def main():
    # 1) Generate synthetic data
    np.random.seed(42)
    N = 1000
    df = pd.DataFrame({
        "Age": np.random.randint(18, 70, N),
        "Income": np.random.normal(60000, 15000, N),
        "Debt": np.random.normal(10000, 5000, N),
        "Credit_Score": np.random.randint(300, 850, N),
        "Years_Employed": np.random.randint(0, 40, N),
        "Num_Accounts": np.random.randint(1, 10, N),
    })
    df["Approved"] = (
        (df["Credit_Score"] > 600) &
        (df["Debt"] / (df["Income"] + 1) < 0.5)
    ).astype(int)

    # 2) Train/test split
    X = df.drop("Approved", axis=1)
    y = df["Approved"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3) Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    print(f"Train acc: {clf.score(X_train,y_train):.3f}, "
          f"Test acc: {clf.score(X_test,y_test):.3f}")

    # 4) SHAP explanations
    explainer = shap.Explainer(clf, X_train, seed=42)
    shap_exp = explainer(X_test)            # (200,6,2)
    shap_pos = shap_exp.values[:,:,1]       # positive class slice

    # 5) Global summary plot
    plt.figure(figsize=(8,6))
    shap.summary_plot(shap_pos, X_test, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150)
    plt.close()
    print("Saved shap_summary.png")

    # 6) Local force plot for sample #0
    shap.initjs()
    base = explainer.expected_value[1]
    vals0 = shap_pos[0]
    feats0 = X_test.iloc[0]
    fig = shap.plots.force(
        base, vals0, feats0,
        feature_names=X_test.columns,
        matplotlib=False
    )
    shap.save_html("force_plot.html", fig)
    print("Saved force_plot.html")

if __name__ == "__main__":
    main()
