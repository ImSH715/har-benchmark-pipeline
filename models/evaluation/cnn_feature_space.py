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

# ==========================================
# 1. DEVICE & MODEL SETUP (train_cnn.py와 동일해야 함)
# ==========================================
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
        x = self.fc1(x)       # ← 이 벡터를 뽑을 거야
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 학습된 가중치 로드
model = CNN1D().to(device)

# 네가 저장한 모델 파일 경로로 바꿔
model_path = '../../saved_models/cnn1d_20260629_154423.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"Loaded model: {model_path}")

# ==========================================
# 2. LOAD DATA
# ==========================================
BASE_DIR = '../../Dataset/Preprocessed/for_dl'
META_DIR = '../../Dataset/Preprocessed/metadata'

# Train 데이터로 임베딩 추출 (전체 다 하면 느리니 3000~5000개 샘플링)
X = np.load(f'{BASE_DIR}/X_train.npy')   # (N, 128, 6)
y = np.load(f'{BASE_DIR}/y_train.npy') - 1  # 0-based

meta = pd.read_csv(f'{META_DIR}/meta_train.csv')
dataset_names = meta['dataset_name'].values

# 샘플링 (t-SNE 속도 때문에)
sample_size = 17867
np.random.seed(42)
idx = np.random.choice(len(X), size=min(sample_size, len(X)), replace=False)

X_sample = X[idx]
y_sample = y[idx]
ds_sample = dataset_names[idx]

# PyTorch 텐서로 변환 + permute
X_t = torch.FloatTensor(X_sample).permute(0, 2, 1).to(device)

# ==========================================
# 3. EXTRACT EMBEDDINGS (FC1 출력)
# ==========================================
print("\nExtracting embeddings...")

embeddings = []
predictions = []

with torch.no_grad():
    for i in range(0, len(X_t), 64):  # batch 처리
        batch = X_t[i:i+64]
        
        # 임베딩 뽑기: conv까지는 동일하게 통과, fc1까지만
        x = model.pool1(model.relu1(model.conv1(batch)))
        x = model.pool2(model.relu2(model.conv2(x)))
        x = x.view(x.size(0), -1)
        emb = model.fc1(x)  # ← 256차원 임베딩
        
        embeddings.append(emb.cpu().numpy())
        
        # 예측값도 같이 저장 (학습 후 분류 결과 확인용)
        out = model.fc2(model.dropout(emb))
        pred = torch.argmax(out, dim=1)
        predictions.append(pred.cpu().numpy())

embeddings = np.vstack(embeddings)      # (5000, 256)
predictions = np.concatenate(predictions)  # (5000,)

print(f"Embeddings shape: {embeddings.shape}")

# ==========================================
# 4. DIMENSIONALITY REDUCTION
# ==========================================
print("\nRunning PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(embeddings)
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

print("\nRunning t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(embeddings)

# ==========================================
# 5. PLOT FUNCTION
# ==========================================
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

# ==========================================
# 6. PLOT 1: TRUE LABEL (정답 기준)
# ==========================================
plot_2d(
    X_tsne,
    y_sample,
    title='CNN Embedding (t-SNE) - Colored by TRUE LABEL',
    save_path='figures/cnn_tsne_true_label.png'
)

# ==========================================
# 7. PLOT 2: PREDICTED LABEL (모델 예측 기준)
# ==========================================
plot_2d(
    X_tsne,
    predictions,
    title='CNN Embedding (t-SNE) - Colored by MODEL PREDICTION',
    save_path='figures/cnn_tsne_pred_label.png'
)

# ==========================================
# 8. PLOT 3: DATASET SOURCE
# ==========================================
ds_map = {name: i for i, name in enumerate(np.unique(ds_sample))}
ds_numeric = np.array([ds_map[name] for name in ds_sample])

plot_2d(
    X_tsne,
    ds_numeric,
    title='CNN Embedding (t-SNE) - Colored by DATASET SOURCE',
    cmap='Set1',
    save_path='figures/cnn_tsne_dataset.png'
)

# ==========================================
# 9. PLOT 4: PCA (전체 구조)
# ==========================================
plot_2d(
    X_pca,
    y_sample,
    title='CNN Embedding (PCA) - Colored by TRUE LABEL',
    save_path='figures/cnn_pca_true_label.png'
)

print("\nDone")