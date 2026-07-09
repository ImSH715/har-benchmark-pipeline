import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

# Load
emb = np.load('../embeddings/cnn_lstm_emb_test.npy')
y_test = np.load('../Dataset/Preprocessed/for_dl/y_test.npy')

# Quick K-means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=11, random_state=42, n_init=10).fit(emb)
pred = kmeans.predict(emb)

# Check basic stats
print(f"y_test range: {y_test.min()} ~ {y_test.max()}")
print(f"pred range:   {pred.min()} ~ {pred.max()}")
print(f"\ny_test first 20: {y_test[:20]}")
print(f"pred first 20:   {pred[:20]}")

# Manual CM check
true_labels = sorted(np.unique(y_test))
cm = confusion_matrix(y_test, pred, labels=true_labels)
print(f"\nCM shape: {cm.shape}")
print(f"CM diagonal (correct if mapped perfectly): {np.diag(cm)}")
print(f"Best possible accuracy (sum of max per row / total): {np.sum(cm.max(axis=1)) / cm.sum():.4f}")

# Hungarian mapping
row_ind, col_ind = linear_sum_assignment(-cm)
mapping = {cluster_id: true_labels[row_idx] 
           for row_idx, cluster_id in zip(row_ind, col_ind)}
mapped = np.array([mapping.get(c, -1) for c in pred])

print(f"\nMapped first 20: {mapped[:20]}")
print(f"Match first 20:  {y_test[:20] == mapped[:20]}")
print(f"Total correct: {(y_test == mapped).sum()} / {len(y_test)}")