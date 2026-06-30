import pandas as pd
import numpy as np
from sklearn.svm import SVC
#Linear SVC
from sklearn.svm import LinearSVC
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

df = pd.read_csv(f'{BASE_DIR}/X_train_features.csv')
print(df.head())
print(df.describe())
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
clf = LinearSVC(
    #kernel='rbf', # Radial Basis Function
    C = 1.0,
    #gamma = 'scale',
    #decision_function_shape='ovo',
    class_weight='balanced',
    max_iter=5000,
    random_state=43
)
print("\nTraining")

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
model_path = f'{model_dir}/svm_{timestamp}.pkl'
results_path = f'{results_dir}/svm_{timestamp}.json'
with open(model_path, 'wb') as f:
    pickle.dump(clf, f)
print(f"\nModel saved: {model_path}")

# Save evaluation results
results = {
    'model': 'SVM_RBF',
    'timestamp': timestamp,
    'kernel': 'rbf',
    'gamma': 'scale',
    'val_accuracy': float(val_acc),
    'test_accuracy': float(test_acc),
    'classification_report': classification_report(y_test, test_pred, output_dict=True)
}
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved : {results_path}")
print(f"Saved : {results_path}")

print("Done")