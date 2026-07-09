"""
Unsupervised Clustering on CNN-LSTM Embeddings
Train: K-means on extracted embeddings (no labels)
Test: Evaluate against ground truth via Hungarian matching
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
print("Unsupervised K-means on CNN-LSTM Embeddings")
print("=" * 60)

# ==========================================
# 1. LOAD EMBEDDINGS & LABELS
# ==========================================
EMB_DIR = '../embeddings'
LABEL_DIR = '../Dataset/Preprocessed/for_dl'

# Embeddings extracted from CNN-LSTM (before final FC layer)
#X_train = np.load(f'{EMB_DIR}/cnn_lstm_emb_train.npy')
#X_test  = np.load(f'{EMB_DIR}/cnn_lstm_emb_test.npy')
X_train = np.load(f'{EMB_DIR}/simclr_emb_train.npy')
X_test  = np.load(f'{EMB_DIR}/simclr_emb_test.npy')
# Labels are used ONLY for final evaluation (never during clustering)
y_train = np.load(f'{LABEL_DIR}/y_train.npy')  # 1-based original labels
y_test  = np.load(f'{LABEL_DIR}/y_test.npy')

print(f"Train embeddings: {X_train.shape}")
print(f"Test  embeddings: {X_test.shape}")
print(f"Unique labels in test: {sorted(np.unique(y_test))}")

# ==========================================
# 2. OPTIONAL PREPROCESSING
# ==========================================
USE_PCA = False  # Set to True if you want to reduce dimensions

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

if USE_PCA:
    pca = PCA(n_components=0.95)
    X_train_proc = pca.fit_transform(X_train_scaled)
    X_test_proc  = pca.transform(X_test_scaled)
    print(f"\nPCA enabled: reduced to {X_train_proc.shape[1]} components")
else:
    X_train_proc = X_train_scaled
    X_test_proc  = X_test_scaled
    print("\nUsing scaled embeddings directly (no PCA)")

# ==========================================
# 3. K-MEANS CLUSTERING (unsupervised)
# ==========================================
n_clusters = len(np.unique(y_train))
print(f"\nTraining K-means with k={n_clusters} ...")

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(X_train_proc)

cluster_pred = kmeans.predict(X_test_proc)

# ==========================================
# 4. HUNGARIAN MATCHING (cluster -> true label)
# ==========================================
# CRITICAL FIX: Convert y_test to 0-based so that both y_test and cluster_pred
# are in the SAME range (0~10). This avoids confusion_matrix shape mismatch
# between 1-based labels and 0-based cluster ids.
y_test_0based = y_test - 1  # 1~11 -> 0~10

# Build confusion matrix with both in 0~10 range -> guaranteed (11, 11)
cm = confusion_matrix(y_test_0based, cluster_pred)

# Hungarian algorithm: rows = true labels (0~10), cols = clusters (0~10)
row_ind, col_ind = linear_sum_assignment(-cm)

# mapping: {cluster_id: true_label_0based}
mapping = {int(col): int(row) for row, col in zip(row_ind, col_ind)}

label_names = {
    0: 'WALKING', 1: 'WALKING_UPSTAIRS', 2: 'WALKING_DOWNSTAIRS',
    3: 'SITTING', 4: 'STANDING', 5: 'LAYING',
    6: 'RUNNING', 7: 'SHUFFLING', 8: 'PICKING',
    9: 'JUMPING', 10: 'CYCLING'
}
print("\nCluster -> Label Mapping:")
for c, l in sorted(mapping.items()):
    print(f"  Cluster {c:2d} -> Label {l+1:2d} ({label_names.get(l, '?')})")

# Apply mapping and convert back to 1-based labels
mapped_pred_0based = np.array([mapping.get(int(c), -1) for c in cluster_pred])
mapped_pred = mapped_pred_0based + 1  # back to 1~11

# ==========================================
# 5. EVALUATION (F1-score against ground truth)
# ==========================================
acc = accuracy_score(y_test, mapped_pred)
print(f"\n{'='*60}")
print(f"Accuracy: {acc:.4f}")
print(f"{'='*60}")

target_names = [label_names.get(lbl-1, f'Class_{lbl}') for lbl in sorted(np.unique(y_test))]
print("\nClassification Report:")
print(classification_report(
    y_test, mapped_pred,
    labels=sorted(np.unique(y_test)),
    target_names=target_names
))

# ==========================================
# 6. CONFUSION MATRIX PLOT
# ==========================================
plt.figure(figsize=(13, 10))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=[f'Cluster {i}' for i in range(n_clusters)],
    yticklabels=[label_names.get(i, f'Class_{i+1}') for i in range(n_clusters)]
)
plt.title('Confusion Matrix: True Label vs K-means Cluster\n(CNN-LSTM Embeddings)', fontsize=14)
plt.ylabel('True Label')
plt.xlabel('K-means Cluster')
plt.tight_layout()

os.makedirs('figures', exist_ok=True)
plt.savefig('figures/kmeans_cnn_lstm_embedding.png', dpi=300)
print("\nSaved: figures/kmeans_cnn_lstm_embedding.png")
plt.show()