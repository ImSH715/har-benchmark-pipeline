import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import pickle
import os
from datetime import datetime

# Loading Directory
BASE_DIR = '../../../DATASET/Preprocessed/for_ml'
model_dir = 'models/saved_models'
results_dir = 'models/results'

os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

X_train = pd.read_csv(f'{BASE_DIR}/X_train_features.csv').values
X_val = pd.read_csv(f'{BASE_DIR}/X_val_features.csv').values
X_test = pd.read_csv(f'{BASE_DIR}/X_test_features.csv').values

y_train = np.load(f'{BASE_DIR}/y_train.npy')
y_test = np.load(f'{BASE_DIR}/y_test.npy')
y_val = np.load(f'{BASE_DIR}/y_val.npy')

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    random_state=43,
    n_jobs=-1,
    class_weight='balanced'
)

# Training
clf.fit(X_train, y_train)

# Evaluation
val_pred = clf.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)

test_pred = clf.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)

print(f"\nVal Accuracy : {val_acc:.4f}")
print(f"\nTest Accuracy : {test_acc:.4f}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f'{model_dir}/rf_{timestamp}.pkl'
results_path = f'{results_dir}/rf_{timestamp}.json'
with open(model_path, 'wb') as f:
    pickle.dump(clf, f)
print(f"\nModel saved: {model_path}")

# Save evaluation results
results = {
    'model': 'RandomForest',
    'timestamp': timestamp,
    'n_estimators': 200,
    'min_samples_split': 5,
    'val_accuracy': float(val_acc),
    'test_accuracy': float(test_acc),
    'classification_report': classification_report(y_test, test_pred, output_dict=True)
}
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved : {results_path}")

# Feature Importance
importance = clf.feature_importances_
feat_names = pd.read_csv(f'{BASE_DIR}/X_train_features.csv').columns
top_idx = np.argsort(importance)[-10:][::-1]

for i  in top_idx:
    print(f"{feat_names[i]}: {importance[i]:.4f}")

print("Done")