import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

# Load
emb = np.load('../embeddings/cnn_lstm_emb_test.npy')
y_test = np.load('../Dataset/Preprocessed/for_dl/y_test.npy')

# K-means
kmeans = KMeans(n_clusters=11, random_state=42, n_init=10).fit(emb)
pred = kmeans.predict(emb)

# Confusion matrix WITH and WITHOUT labels parameter
true_labels = sorted(np.unique(y_test))

print("=== WITHOUT labels parameter ===")
cm_auto = confusion_matrix(y_test, pred)
print(f"Shape: {cm_auto.shape}")
print(cm_auto[:3, :3])  # Top-left corner

print("\n=== WITH labels parameter ===")
cm_labeled = confusion_matrix(y_test, pred, labels=true_labels)
print(f"Shape: {cm_labeled.shape}")
print(cm_labeled[:3, :3])

# Hungarian on both
for name, cm in [("Auto", cm_auto), ("Labeled", cm_labeled)]:
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    print(f"\n=== {name} Mapping ===")
    print(f"row_ind: {row_ind}")
    print(f"col_ind: {col_ind}")
    
    # Check if mapping makes sense
    mapping = {int(col): int(row) for row, col in zip(row_ind, col_ind)}
    print(f"mapping (col->row): {mapping}")
    
    # Apply and check accuracy
    mapped = np.array([mapping.get(int(c), -1) for c in pred])
    
    # For Auto: row index = true label index (if sorted), but we need actual label values
    # For Labeled: row index maps to true_labels[row_idx]
    if name == "Labeled":
        mapped = np.array([true_labels[mapping.get(int(c), 0)] for c in pred])
    
    acc = (y_test == mapped).sum() / len(y_test)
    print(f"Accuracy: {acc:.4f}")