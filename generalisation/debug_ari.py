import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

emb = np.load('../embeddings/cnn_lstm_emb_test.npy')
y_test = np.load('../Dataset/Preprocessed/for_dl/y_test.npy')

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=11, random_state=42, n_init=10).fit(emb)
pred = kmeans.predict(emb)

print(f"ARI: {adjusted_rand_score(y_test, pred):.4f}")
print(f"NMI: {normalized_mutual_info_score(y_test, pred):.4f}")