import os
import pandas as pd

RESULTS_ROOT = "/home/onur/Desktop/Project/proj/src/sae/results"
OUTPUT_CSV = os.path.join(RESULTS_ROOT, "summary_metrics.csv")

summary_rows = []

for root, dirs, files in os.walk(RESULTS_ROOT):
    if "metrics.csv" in files:
        metrics_path = os.path.join(root, "metrics.csv")
        parts = root.split(os.sep)

        # Try to find parts like 'layer16', 'dim2048', 'k16', 'lr1e-04' anywhere in the path
        try:
            layer = int(next(p.replace("layer", "") for p in parts if p.startswith("layer")))
            d_hidden = int(next(p.replace("dim", "") for p in parts if p.startswith("dim")))
            k_val = int(next(p.replace("k", "") for p in parts if p.startswith("k")))
            lr_str = next(p.replace("lr", "") for p in parts if p.startswith("lr"))
            lr = float(lr_str)
        except Exception as e:
            print(f"[SKIP] Failed to parse path: {metrics_path} | Error: {e}")
            continue

        df = pd.read_csv(metrics_path)
        if len(df) == 0:
            continue
        last = df.iloc[-1]

        summary_rows.append({
            "layer": layer,
            "k": k_val,
            "d_hidden": d_hidden,
            "lr": lr,
            "epoch": last.get("epoch", None),
            "train_loss": last.get("train_loss_step", None),
            "mse_loss": last.get("train_mse_loss_step", None),
            "auxk_loss": last.get("train_auxk_loss_step", None),
            "num_dead_neurons": last.get("num_dead_neurons_step", None)
        })

if not summary_rows:
    print("[ERROR] No valid metric files found or parsed.")
else:
    summary_df = pd.DataFrame(summary_rows)
    summary_df.sort_values(by=["layer", "d_hidden", "k", "lr"], inplace=True)
    summary_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[DONE] Saved summary to {OUTPUT_CSV}")
