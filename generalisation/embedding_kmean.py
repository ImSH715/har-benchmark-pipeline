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
X_train = np.load(f'{EMB_DIR}/cnn_lstm_emb_train.npy')
X_test  = np.load(f'{EMB_DIR}/cnn_lstm_emb_test.npy')

# Labels are used ONLY for final evaluation (never during clustering)
y_train = np.load(f'{LABEL_DIR}/y_train.npy')  # 1-based
y_test  = np.load(f'{LABEL_DIR}/y_test.npy')

print(f"Train embeddings: {X_train.shape}")
print(f"Test  embeddings: {X_test.shape}")
print(f"Unique labels in test: {sorted(np.unique(y_test))}")

# ==========================================
# 2. OPTIONAL PREPROCESSING
# Embeddings are already learned representations, but scaling helps K-means.
# PCA is optional; you can comment it out to use raw 128-dim embeddings.
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
# Number of clusters = number of unique true classes
n_clusters = len(np.unique(y_train))
print(f"\nTraining K-means with k={n_clusters} ...")

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(X_train_proc)  # <-- NO labels used here

# Predict clusters on test set
cluster_pred = kmeans.predict(X_test_proc)

# ==========================================
# 4. HUNGARIAN MATCHING (cluster -> true label)
# ==========================================
# Get sorted list of actual label values present in the dataset
true_labels = sorted(np.unique(y_test))

# Build confusion matrix between true labels and cluster IDs.
# We pass 'labels=true_labels' so each row corresponds to an actual label value.
cm = confusion_matrix(y_test, cluster_pred, labels=true_labels)

# Hungarian algorithm finds the optimal one-to-one assignment
# between rows (true labels) and columns (clusters) by maximizing overlap.
row_ind, col_ind = linear_sum_assignment(-cm)

# row_ind contains matrix ROW INDICES (0, 1, 2...), NOT the actual label values.
# We index into 'true_labels' to recover the real label.
mapping = {cluster_id: true_labels[row_idx]
           for row_idx, cluster_id in zip(row_ind, col_ind)}

# Pretty-print mapping
label_names = {
    1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING', 5: 'STANDING', 6: 'LAYING',
    7: 'RUNNING', 8: 'SHUFFLING', 9: 'PICKING',
    10: 'JUMPING', 11: 'CYCLING'
}
print("\nCluster -> Label Mapping:")
for c, l in sorted(mapping.items()):
    print(f"  Cluster {c:2d} -> Label {l:2d} ({label_names.get(l, '?')})")

# Convert cluster IDs to predicted labels using the mapping
mapped_pred = np.array([mapping.get(c, -1) for c in cluster_pred])

# ==========================================
# 5. EVALUATION (F1-score against ground truth)
# ==========================================
acc = accuracy_score(y_test, mapped_pred)
print(f"\n{'='*60}")
print(f"Accuracy: {acc:.4f}")
print(f"{'='*60}")

target_names = [label_names.get(lbl, f'Class_{lbl}') for lbl in true_labels]
print("\nClassification Report:")
print(classification_report(
    y_test, mapped_pred,
    labels=true_labels,
    target_names=target_names
))

# ==========================================
# 6. CONFUSION MATRIX PLOT
# ==========================================
plt.figure(figsize=(13, 10))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=[f'Cluster {i}' for i in range(n_clusters)],
    yticklabels=target_names
)
plt.title('Confusion Matrix: True Label vs K-means Cluster\n(CNN-LSTM Embeddings)', fontsize=14)
plt.ylabel('True Label')
plt.xlabel('K-means Cluster')
plt.tight_layout()

os.makedirs('figures', exist_ok=True)
plt.savefig('figures/kmeans_cnn_lstm_embedding.png', dpi=300)
print("\nSaved: figures/kmeans_cnn_lstm_embedding.png")
plt.show()
