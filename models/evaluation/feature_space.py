"""
Feature Space Visualization using t-SNE and PCA
Input: ML features (N, 54) + labels + dataset_name
Output: 2D scatter plots
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

print("=" * 60)
print("Feature Space Visualization")
print("=" * 60)

# LOAD DATA
BASE_DIR = '../../Dataset/Preprocessed/for_ml'
META_DIR = '../../Dataset/Preprocessed/metadata'

# 피처
X = pd.read_csv(f'{BASE_DIR}/X_train_features.csv').values

# metadata (dataset_name, label)
meta = pd.read_csv(f'{META_DIR}/meta_train.csv')

dataset_names = meta['dataset_name'].values
labels = meta['label'].values

print(f"Features shape: {X.shape}")
print(f"Datasets: {np.unique(dataset_names)}")
print(f"Labels: {sorted(np.unique(labels))}")

# DIMENSIONALITY REDUCTION
# PCA
print("\nRunning PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

# t-sne
sample_size = 17867 # all samples
np.random.seed(42)
idx = np.random.choice(len(X), size=min(sample_size, len(X)), replace=False)

X_sample = X[idx]
labels_sample = labels[idx]
dataset_sample = dataset_names[idx]

print(f"\nRunning t-SNE on {len(idx)} samples...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_sample)

# PLOT FUNCTION
def plot_2d(X_2d, color_vals, title, cmap='tab10', save_path=None):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=color_vals, cmap=cmap, alpha=0.6, s=10)
    plt.colorbar(scatter, label='Class' if 'Class' in title else 'Dataset')
    plt.title(title, fontsize=14)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

# PLOT BY CLASS LABEL (t-SNE)
plot_2d(
    X_tsne, 
    labels_sample, 
    title=f't-SNE by Activity Class (n={len(idx)})',
    save_path='figures/tsne_by_class.png'
)

# PLOT BY DATASET SOURCE (t-SNE)
# map dataset_name with int
ds_map = {name: i for i, name in enumerate(np.unique(dataset_sample))}
ds_numeric = np.array([ds_map[name] for name in dataset_sample])

plot_2d(
    X_tsne,
    ds_numeric,
    title=f't-SNE by Dataset Source (n={len(idx)})',
    cmap='Set1',
    save_path='figures/tsne_by_dataset.png'
)

# PLOT BY CLASS
plot_2d(
    X_pca,
    labels,
    title='PCA by Activity Class (All Data)',
    save_path='figures/pca_by_class.png'
)

# PLOT BY DATASET
ds_map_full = {name: i for i, name in enumerate(np.unique(dataset_names))}
ds_numeric_full = np.array([ds_map_full[name] for name in dataset_names])

plot_2d(
    X_pca,
    ds_numeric_full,
    title='PCA by Dataset Source (All Data)',
    cmap='Set1',
    save_path='figures/pca_by_dataset.png'
)

print("\n" + "=" * 60)
print("Visualization Complete")
print("=" * 60)