import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_regional_metrics(pred_csv_path):
    print(f"Analyzing regions for: {pred_csv_path}...")
    df = pd.read_csv(pred_csv_path)

    rows = []
    group_cols = ["region", "parameter", "model"]

    for (region, parameter, model), group in df.groupby(group_cols):
        y_true = group["y_true"].values
        y_pred = group["y_pred"].values

        if len(y_true) < 2:
            r2 = float("nan")
        else:
            r2 = float(r2_score(y_true, y_pred))

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))

        rows.append({
            "region": region,
            "parameter": parameter,
            "model": model,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "n_samples": len(y_true),
        })

    out_path = Path("supplementary/regional_performance/regional_rmse_by_parameter.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
    print(f" -> Saved regional metrics to {out_path}")

def plot_error_distributions(pred_csv_path):
    df = pd.read_csv(pred_csv_path)
    model_name = df["model"].iloc[0] if not df.empty else "unknown"
    out_dir = Path("supplementary/error_analysis/histograms")
    out_dir.mkdir(parents=True, exist_ok=True)

    for parameter, group in df.groupby("parameter"):
        residuals = group["y_pred"] - group["y_true"]

        plt.figure(figsize=(6,4))
        plt.hist(residuals, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
        plt.title(f"Residuals: {parameter} ({model_name})")
        plt.xlabel("Prediction Error")
        plt.ylabel("Frequency")
        plt.grid(axis="y", alpha=0.3)

        fname = f"{model_name}_{parameter}_resid.png".replace(" ", "_")
        plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()
    print(f" -> Histograms saved to {out_dir}")
