import os
import pandas as pd


# Path of root folder where all hyperparameters from grid seach outputs
# Output parh of csv 
RESULTS_ROOT = "proj/grid_results"
OUTPUT_CSV = os.path.join(RESULTS_ROOT, "top_latents_summary.csv")


# Scans training results directories and extracts top 10 latent neuron indices per class (from csv file class_0_weights.csv)
# Parses metada from foldernames (layer, k, hidden dim, learning rate)
# Save summary table of the top neurons per class into one csv file




summary_rows = []

# Recursively checks all subdirectories and files under proj/grid_results
for root, dirs, files in os.walk(RESULTS_ROOT):
    for file in files:
        if file.startswith("class_") and file.endswith("_weights.csv"):         # Looks for files like class_0_weights.csv 
            weights_path = os.path.join(root, file)
            try:
                df = pd.read_csv(weights_path)
                top_indices = df.head(10)["Index"].tolist()  # top 10 neurons  
                
                # Parses metadata from folder hierarchy such that proj/grid_results/layer16/dim2048/k128/lr1e-04/class_0_weights.csv 
                parts = root.split(os.sep)
                lr_str = parts[-2].replace("lr", "")
                lr = float(lr_str.replace("e-", "e-") if "e-" in lr_str else lr_str)
                k_val = int(parts[-3].replace("k", ""))
                d_hidden = int(parts[-4].replace("dim", ""))
                layer = int(parts[-5].replace("layer", ""))
                class_id = int(file.split("_")[1])

                # Builds summary 
                summary_rows.append({
                    "layer": layer,
                    "k": k_val,
                    "d_hidden": d_hidden,
                    "lr": lr,
                    "class_id": class_id,
                    "top_latent_indices": ",".join(map(str, top_indices))
                })
            except Exception as e:
                print(f"[WARN] Failed to process {weights_path}: {e}")
                continue

# Saves as dataframe 
summary_df = pd.DataFrame(summary_rows)
summary_df.sort_values(by=["layer", "d_hidden", "k", "lr", "class_id"], inplace=True)

#Output top_latents+summary.csv 
summary_df.to_csv(OUTPUT_CSV, index=False)

print(f"[DONE] Saved top latent neuron summary to {OUTPUT_CSV}")
