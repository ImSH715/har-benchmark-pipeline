import pandas as pd
import glob
import numpy as np
import os
import json

# Load config.json file

with open('config.json', 'r') as f:
    config = json.load(f)
DATASET_CONFIG = config['datasets']
print(f"Loaded config with {len(DATASET_CONFIG)} datasets")

# Adding metadata
def add_metadata(df, dataset_config):
    df.insert(0, 'dataset_name', dataset_config['dataset_name'])
    df.insert(1, 'subject_id', pd.NA)
    df.insert(2, 'age', dataset_config['default_age'])
    df.insert(3, 'gender', dataset_config['default_gender'])
    df.insert(4, 'condition', dataset_config['default_condition'])
    return df
# Combining
all_dfs = []
for key, dataset_info in DATASET_CONFIG.items():
    print(f"Processing {key}")
    df = pd.read_csv(dataset_info['path'])
    df = add_metadata(df, dataset_info)
    all_dfs.append(df)

print("Merging")
combined_df = pd.concat(all_dfs, ignore_index=True)

# Save
out_path = 'Dataset/Preprocessed/Combined/combined_dataset.csv'
os.makedirs('Dataset/Preprocessed/Combined', exist_ok=True)
combined_df.to_csv(out_path, index=False)

print(f"Saved {out_path}")
