"""
Visualize CNN learned embeddings using t-SNE/PCA
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

# DEVICE & MODEL SETUP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN1D(nn.Module):
    def __init__(self, num_classes=11):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(6, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        
        self.fc1 = nn.Linear(128 * 32, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)       # Vector extraction
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN1D().to(device)

# Saved model
model_path = '../../saved_models/cnn1d_20260629_154423.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"Loaded model: {model_path}")

# LOAD DATA
BASE_DIR = '../../Dataset/Preprocessed/for_dl'
META_DIR = '../../Dataset/Preprocessed/metadata'

# Embedding extraction using Train data
X = np.load(f'{BASE_DIR}/X_train.npy') 
y = np.load(f'{BASE_DIR}/y_train.npy') - 1

meta = pd.read_csv(f'{META_DIR}/meta_train.csv')
dataset_names = meta['dataset_name'].values

# Sampling size
sample_size = 17867 # all samples
np.random.seed(42)
idx = np.random.choice(len(X), size=min(sample_size, len(X)), replace=False)

X_sample = X[idx]
y_sample = y[idx]
ds_sample = dataset_names[idx]

# Convert to pyTorch tensor
X_t = torch.FloatTensor(X_sample).permute(0, 2, 1).to(device)

# EXTRACT EMBEDDINGS
print("\nExtracting embeddings...")

embeddings = []
predictions = []

with torch.no_grad():
    for i in range(0, len(X_t), 64):  # batch
        batch = X_t[i:i+64]
        
        # Embedding extraction
        x = model.pool1(model.relu1(model.conv1(batch)))
        x = model.pool2(model.relu2(model.conv2(x)))
        x = x.view(x.size(0), -1)
        emb = model.fc1(x)  # ← 256차원 임베딩
        
        embeddings.append(emb.cpu().numpy())
        
        out = model.fc2(model.dropout(emb))
        pred = torch.argmax(out, dim=1)
        predictions.append(pred.cpu().numpy())

embeddings = np.vstack(embeddings)     
predictions = np.concatenate(predictions)

print(f"Embeddings shape: {embeddings.shape}")

# DIMENSIONALITY REDUCTION
print("\nRunning PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(embeddings)
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

print("\nRunning t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(embeddings)

# PLOT FUNCTION
def plot_2d(X_2d, color_vals, title, cmap='tab10', save_path=None):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=color_vals, cmap=cmap, alpha=0.6, s=10)
    plt.colorbar(scatter)
    plt.title(title, fontsize=14)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

# PLOT 1: TRUE LABEL : based on the answers
plot_2d(
    X_tsne,
    y_sample,
    title='CNN Embedding (t-SNE) - Colored by TRUE LABEL',
    save_path='figures/cnn_tsne_true_label.png'
)

# PLOT 2: PREDICTED LABEL : based on the model's prediction
plot_2d(
    X_tsne,
    predictions,
    title='CNN Embedding (t-SNE) - Colored by MODEL PREDICTION',
    save_path='figures/cnn_tsne_pred_label.png'
)

# PLOT 3: DATASET SOURCE
ds_map = {name: i for i, name in enumerate(np.unique(ds_sample))}
ds_numeric = np.array([ds_map[name] for name in ds_sample])

plot_2d(
    X_tsne,
    ds_numeric,
    title='CNN Embedding (t-SNE) - Colored by DATASET SOURCE',
    cmap='Set1',
    save_path='figures/cnn_tsne_dataset.png'
)

# PLOT 4: PCA 
plot_2d(
    X_pca,
    y_sample,
    title='CNN Embedding (PCA) - Colored by TRUE LABEL',
    save_path='figures/cnn_pca_true_label.png'
)

print("\nDone")