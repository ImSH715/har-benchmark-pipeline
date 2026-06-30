import pandas as pd
import numpy as np
import os
import json

print("=" * 60)
print("LABEL UNIFICATION")
print("Mapping all datasets to combined.txt standard")
print("=" * 60)

# Load label.json
with open('label.json', 'r') as f:
    config = json.load(f)

LABEL_MAP = {}
for dataset_name, mapping_dict in config['datasets'].items():
    # JSON keys are strings, convert to int for label mapping
    LABEL_MAP[dataset_name] = {int(k): int(v) for k, v in mapping_dict.items()}

print(f"Loaded label mappings for {len(LABEL_MAP)} datasets")

# Paths
INPUT_DIR = 'Dataset/Preprocessed/Windowed'
OUTPUT_DIR = 'Dataset/Preprocessed/label_unified'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# PROCESS EACH DATASET
datasets = {
    'UCI_HAR': 'UCI_HAR_50Hz_2.56s.csv',
    'KU_HAR': 'KU-HAR_50Hz_2.56s.csv',
    'HARTH': 'HARTH_50Hz_2.56s.csv',
    'RealDISP': 'RealDISP_50Hz_2.56s.csv',
}

for dataset_name, filename in datasets.items():
    input_path = os.path.join(INPUT_DIR, filename)
    
    if not os.path.exists(input_path):
        print(f"\n[SKIP] {input_path} not found. Please copy Windowed files first.")
        continue
    
    print(f"\n[INFO] Processing {dataset_name}...")
    df = pd.read_csv(input_path)
    
    original_count = len(df)
    print(f"  Original rows: {original_count}")
    print(f"  Original labels: {sorted(df['label'].unique())}")
    
    # Apply mapping
    mapping = LABEL_MAP[dataset_name]
    df['unified_label'] = df['label'].map(mapping)
    
    # Filter: keep only rows with valid unified label
    df_filtered = df.dropna(subset=['unified_label']).copy()
    df_filtered['unified_label'] = df_filtered['unified_label'].astype(int)
    
    # Drop original label, rename unified to label
    df_filtered = df_filtered.drop(columns=['label'])
    df_filtered = df_filtered.rename(columns={'unified_label': 'label'})
    
    filtered_count = len(df_filtered)
    removed_count = original_count - filtered_count
    print(f"  Filtered rows: {filtered_count} (removed {removed_count})")
    print(f"  Unified labels: {sorted(df_filtered['label'].unique())}")
    
    # Save
    output_filename = f"{dataset_name}_unified.csv"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    df_filtered.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
