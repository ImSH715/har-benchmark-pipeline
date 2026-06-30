import pandas as pd
import numpy as np
#xgboost
from xgboost import XGBClassifier
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

# 0-based conversion
y_train_xgb = y_train - 1
y_val_xgb   = y_val - 1
y_test_xgb  = y_test - 1

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
clf = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate = 0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    num_class = 11,
    objective='multi:softprob',
    eval_metric = 'mlogloss',
    random_state = 43,
    n_jobs = -1
)

# Training
clf.fit(X_train, y_train_xgb, eval_set = [(X_val, y_val_xgb)], verbose = False)

# Evaluation
val_pred_xgb = clf.predict(X_val)
test_pred_xgb = clf.predict(X_test)

val_pred = val_pred_xgb + 1
test_pred = test_pred_xgb + 1

val_acc = accuracy_score(y_val, val_pred)
test_acc = accuracy_score(y_test, test_pred)

print(f"\nVal Accuracy : {val_acc:.4f}")
print(f"\nTest Accuracy : {test_acc:.4f}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f'{model_dir}/xgb_{timestamp}.pkl'
results_path = f'{results_dir}/xgb_{timestamp}.json'
with open(model_path, 'wb') as f:
    pickle.dump(clf, f)
print(f"\nModel saved: {model_path}")

# Save evaluation results
results = {
    'model': 'XGBoost',
    'timestamp': timestamp,
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
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