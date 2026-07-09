"""
Unsupervised Clustering Evaluation with StandardScaler + PCA
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 60)
print("Unsupervised Evaluation: K-means + Scaler + PCA")
print("=" * 60)

# ==========================================
# 1. LOAD DATA
# ==========================================
# Change this line only to switch datasets
BASE_DIR = '../../Dataset/Preprocessed/for_ml'  
# BASE_DIR = '../../Dataset/Preprocessed/for_ml_no_shuffle'

X_train = pd.read_csv(f'{BASE_DIR}/X_train_features.csv').values
X_test = pd.read_csv(f'{BASE_DIR}/X_test_features.csv').values
y_train = np.load(f'{BASE_DIR}/y_train.npy')
y_test = np.load(f'{BASE_DIR}/y_test.npy')

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Classes: {sorted(np.unique(y_test))}")

# ==========================================
# 2. PREPROCESSING (Scaler + PCA) BEFORE K-means
# ==========================================
print("\nApplying StandardScaler + PCA...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Keep 95% variance, or cap at 20 components
pca = PCA(n_components=0.95)  
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"PCA reduced to {X_train_pca.shape[1]} components")

# ==========================================
# 3. UNSUPERVISED TRAINING (K-means)
# ==========================================
n_clusters = len(np.unique(y_train))
print(f"\nTraining K-means (k={n_clusters}, no labels)...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(X_train_pca)

# ==========================================
# 4. PREDICT & MAP CLUSTERS
# ==========================================
cluster_pred = kmeans.predict(X_test_pca)
true_labels = sorted(np.unique(y_test))

cm = confusion_matrix(y_test, cluster_pred, labels=true_labels)
row_ind, col_ind = linear_sum_assignment(-cm)

mapping = {cluster_id: true_labels[row_idx] 
           for row_idx, cluster_id in zip(row_ind, col_ind)}

print("\nCluster -> Label Mapping:")
label_names = {
    1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING', 5: 'STANDING', 6: 'LAYING',
    7: 'RUNNING', 8: 'SHUFFLING', 9: 'PICKING',
    10: 'JUMPING', 11: 'CYCLING'
}
for c, l in sorted(mapping.items()):
    print(f"  Cluster {c} -> Label {l} ({label_names.get(l, '?')})")

mapped_pred = np.array([mapping.get(c, -1) for c in cluster_pred])

# ==========================================
# 5. EVALUATE
# ==========================================
acc = accuracy_score(y_test, mapped_pred)
print(f"\nAccuracy: {acc:.4f}")

target_names = [label_names.get(lbl, f'Class_{lbl}') for lbl in true_labels]
print(classification_report(y_test, mapped_pred, 
                            labels=true_labels, 
                            target_names=target_names))

# ==========================================
# 6. PLOT
# ==========================================
plt.figure(figsize=(13, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'Cluster {i}' for i in range(n_clusters)],
            yticklabels=target_names)
plt.title('Confusion Matrix: True Label vs K-means Cluster', fontsize=14)
plt.ylabel('True Label')
plt.xlabel('K-means Cluster')
plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/kmeans_confusion_matrix.png', dpi=300)
print("\nSaved: figures/kmeans_confusion_matrix.png")
plt.show()