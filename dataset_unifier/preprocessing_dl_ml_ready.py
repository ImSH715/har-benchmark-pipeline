"""
A script to preprocess the dataset into final trainable structure.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
import pickle
import os

print("Preprocessing")
print("Train and Test Split + Normalisation + Feature Extraction")


# Path Configuration
BASE_DIR = 'Dataset/Preprocessed'
LABELED_DIR = f'{BASE_DIR}/label_unified'
OUTPUT_DIR = BASE_DIR

# Load Windowed Datasets
datasets = {
    'KU-HAR': pd.read_csv(f"{LABELED_DIR}/KU_HAR_unified.csv"),
    'HARTH': pd.read_csv(f"{LABELED_DIR}/HARTH_unified.csv",),
    'UCI_HAR': pd.read_csv(f"{LABELED_DIR}/UCI_HAR_unified.csv"),
    'RealDISP': pd.read_csv(f"{LABELED_DIR}/RealDISP_unified.csv")
}
# Check Shapes
for name, df in datasets.items():
    print(f"Loade {name}: {df.shape}")

# Add dataset identifier & combine
all_dfs = []
for name, df in datasets.items():
    df_copy = df.copy()
    df_copy.insert(0, 'dataset_name', name)
    all_dfs.append(df_copy)
combined_df = pd.concat(all_dfs, ignore_index=True)
print(f"Combined windowed data: {combined_df.shape}")

# Extract Sensor columns & labels
sensor_cols = [c for c in combined_df.columns if c not in ['dataset_name', 'label']]

X = combined_df[sensor_cols].values
y = combined_df['label'].values
dataset_names = combined_df['dataset_name'].values

# Train/ Val/ Test Split
# Test (15%)
X_temp, X_test, y_temp, y_test, ds_temp, ds_test = train_test_split(X, y, dataset_names, test_size=0.15, random_state=43, stratify=y)
# Val from remaining
X_train, X_val, y_train, y_val, ds_train, ds_val = train_test_split(X_temp, y_temp, ds_temp, test_size=0.176, random_state=43, stratify=y_temp)

print(f"\nTrain: {X_train.shape}")
print(f"Test: {X_test.shape}")
print(f"Val: {X_val.shape}")

# Normalise Raw Data (for deep learning)
scaler_raw = StandardScaler()

# Flatten to Fit on Train to Transform all
X_train_flat = scaler_raw.fit_transform(X_train)
X_val_flat  = scaler_raw.transform(X_val)
X_test_flat = scaler_raw.transform(X_test)

# Reshape back to (N, 128, 6)
X_train_dl = X_train_flat.reshape(-1, 128, 6)
X_val_dl = X_val_flat.reshape(-1, 128, 6)
X_test_dl = X_test_flat.reshape(-1, 128, 6)

print(f"\n DL Train Shape: {X_train_dl.shape}")
print(f"\n DL Test Shape: {X_test_dl.shape}")
print(f"\n DL Val Shape: {X_val_dl.shape}")

# Feature Extraction
FEATURES = ['mean', 'std', 'max', 'min', 'median', 'range', 'rms', 'skew', 'kurtosis']
AXES = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']

def extract_features(X_flat):
    N = X_flat.shape[0]
    X_3d = X_flat.reshape(N, 128, 6)
    feats = []

    for i in range(6):
        axis_data = X_3d[:,:,i]
        feats.append(np.mean(axis_data, axis=1))
        feats.append(np.std(axis_data, axis=1))
        feats.append(np.max(axis_data, axis=1))
        feats.append(np.min(axis_data, axis=1))
        feats.append(np.median(axis_data, axis=1))        
        feats.append(np.max(axis_data, axis=1) - np.min(axis_data, axis=1))
        feats.append(np.sqrt(np.mean(axis_data**2, axis=1)))
        feats.append(skew(axis_data, axis=1))
        feats.append(kurtosis(axis_data, axis=1))

    return np.column_stack(feats)
print("\n Extracting Features")
X_train_feat = extract_features(X_train)
X_test_feat = extract_features(X_test)
X_val_feat = extract_features(X_val)

# Normalising features
scaler_feat = StandardScaler()
X_train_feat = scaler_feat.fit_transform(X_train_feat)
X_val_feat = scaler_feat.transform(X_val_feat)
X_test_feat = scaler_feat.transform(X_test_feat)

print("Train Shape: ", X_train_feat.shape)
print("Test Shape: ", X_test_feat.shape)
print("Val Shape: ", X_val_feat.shape)

# Create output directories & save
for subdir in ['for_dl', 'for_ml', 'metadata', 'scalers']:
    os.makedirs(f'{OUTPUT_DIR}/{subdir}', exist_ok=True)

# For Deep learning
np.save(f'{OUTPUT_DIR}/for_dl/X_train.npy', X_train_dl)
np.save(f'{OUTPUT_DIR}/for_dl/X_test.npy', X_test_dl)
np.save(f'{OUTPUT_DIR}/for_dl/X_val.npy', X_val_dl)
np.save(f'{OUTPUT_DIR}/for_dl/y_train.npy', y_train)
np.save(f'{OUTPUT_DIR}/for_dl/y_test.npy', y_test)
np.save(f'{OUTPUT_DIR}/for_dl/y_val.npy', y_val)
print(f"\nDL saved : {OUTPUT_DIR}/for_dl/")

# For machine learning
feat_col_names = [f"{a}_{f}" for a in AXES for f in FEATURES]
pd.DataFrame(X_train_feat, columns=feat_col_names).to_csv(f'{OUTPUT_DIR}/for_ml/X_train_features.csv', index=False)
pd.DataFrame(X_val_feat, columns=feat_col_names).to_csv(f'{OUTPUT_DIR}/for_ml/X_val_features.csv', index=False)
pd.DataFrame(X_test_feat, columns=feat_col_names).to_csv(f'{OUTPUT_DIR}/for_ml/X_test_features.csv', index=False)
np.save(f'{OUTPUT_DIR}/for_ml/y_train.npy', y_train)
np.save(f'{OUTPUT_DIR}/for_ml/y_val.npy', y_val)
np.save(f'{OUTPUT_DIR}/for_ml/y_test.npy', y_test)
print(f"ML features to {OUTPUT_DIR}/for_ml/")

# Scalers
with open(f'{OUTPUT_DIR}/scalers/scaler_raw.pkl','wb') as f:
    pickle.dump(scaler_raw, f)
with open(f'{OUTPUT_DIR}/scalers/scaler_features.pkl', 'wb') as f:
    pickle.dump(scaler_feat, f)
print(f'Saved Scalers to {OUTPUT_DIR}/scalers')

# Verification
print("Verification")

print(f"\nDL Raw data range (train): {X_train_dl.min():.4f} ~ {X_train_dl.max():.4f}")
print(f"ML Feature range (train): {X_train_feat.min():.4f} ~ {X_train_feat.max():.4f}")

print(f"\nTrain distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Label {int(u)}: {c}")

print(f"\n[Source] Train dataset distribution:")
meta_train = pd.DataFrame({'dataset_name': ds_train, 'label': y_train})
print(meta_train['dataset_name'].value_counts())

meta_train.to_csv(f'{OUTPUT_DIR}/metadata/meta_train.csv', index=False)
meta_test = pd.DataFrame({'dataset_name': ds_test, 'label': y_test})
meta_val = pd.DataFrame({'dataset_name': ds_val, 'label': y_val})
meta_test.to_csv(f'{OUTPUT_DIR}/metadata/meta_test.csv', index=False)
meta_val.to_csv(f'{OUTPUT_DIR}/metadata/meta_val.csv', index=False)

print("PREPROCESSING COMPLETE!")
print(f"\nReady for ML/DL training:")
print(f"  - Random Forest / SVM / XGBoost: use {OUTPUT_DIR}/for_ml/")
print(f"  - CNN / LSTM / CNN-LSTM: use {OUTPUT_DIR}/for_dl/")