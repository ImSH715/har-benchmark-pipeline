import numpy as np

emb = np.load('../embeddings/cnn_lstm_emb_test.npy')
print(f"Shape: {emb.shape}")

print(f"Mean: {emb.mean():.6f}")
print(f"Std:  {emb.std():.6f}")
print(f"Min:  {emb.min():.6f}")
print(f"Max:  {emb.max():.6f}")

print("\nFirst 3 samples:")
print(emb[:3])
