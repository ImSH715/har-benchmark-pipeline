import numpy as np
import os
import pandas as pd

# For ML
ML_DIR = '../../Dataset/Preprocessed/for_ml'
ML_OUT = '../../Dataset/Preprocessed/for_ml_no_shuffle'
os.makedirs(ML_OUT, exist_ok=True)

for split in ['train', 'val', 'test']:
    X = pd.read_csv(f'{ML_DIR}/X_{split}_features.csv').values
    y = np.load(f'{ML_DIR}/y_{split}.npy')

    print(f"\n[ML/{split}] Before: {len(y)} samples, labels: {sorted(np.unique(y))}")

    # Remove class 8 = shuffle

    mask = (y != 8)
    X_filtered = X[mask]
    y_filtered = y[mask]

    print(f"[ML/{split}] After:  {len(y_filtered)} samples, labels: {sorted(np.unique(y_filtered))}")

    #
    pd.DataFrame(X_filtered, columns=pd.read_csv(f'{ML_DIR}/X_train_features.csv').columns).to_csv(
        f'{ML_OUT}/X_{split}_features.csv', index=False)
    np.save(f'{ML_OUT}/y_{split}.npy', y_filtered)

# DL
DL_DIR = '../../Dataset/Preprocessed/for_dl'
DL_OUT = '../../Dataset/Preprocessed/for_dl_no_shuffle'
os.makedirs(DL_OUT, exist_ok=True)

for split in ['train', 'val', 'test']:
    X = np.load(f'{DL_DIR}/X_{split}.npy')
    y = np.load(f'{DL_DIR}/y_{split}.npy')

    print(f"\n[DL/{split}] Before: {len(y)} samples, labels: {sorted(np.unique(y))}")

    mask = (y != 8)
    X_filtered = X[mask]
    y_filtered = y[mask]

    print(f"[DL/{split}] After:  {len(y_filtered)} samples, labels: {sorted(np.unique(y_filtered))}")

    np.save(f'{DL_OUT}/X_{split}.npy', X_filtered)
    np.save(f'{DL_OUT}/y_{split}.npy', y_filtered)

print("Done")
print(f"ML output: {ML_OUT}")
print(f"DL output: {DL_OUT}")
