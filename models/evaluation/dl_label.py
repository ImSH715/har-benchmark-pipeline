"""
Universal Embedding Extractor for CNN / LSTM / CNN_LSTM
Extracts embeddings from the layer just before final classification.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

print("=" * 60)
print("Universal Embedding Extractor (CNN / LSTM / CNN-LSTM)")
print("=" * 60)

# ==========================================
# 1. CONFIGURATION
# ==========================================
# OPTIONS: 'cnn', 'lstm', 'cnn_lstm'
MODEL_TYPE = 'cnn_lstm' 

# File paths
MODEL_PATH = '../../saved_models/cnn_lstm_20260630_142946.pth'
BASE_DIR = '../../Dataset/Preprocessed/for_dl_no_shuffle'
META_DIR = '../../Dataset/Preprocessed/metadata'

# Visualization config
SAMPLE_SIZE = 17867 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Model type: {MODEL_TYPE}")
print(f"Model file: {MODEL_PATH}")

LABEL_NAMES = {
    0: 'WALKING',
    1: 'WALKING_UPSTAIRS',
    2: 'WALKING_DOWNSTAIRS',
    3: 'SITTING',
    4: 'STANDING',
    5: 'LAYING',
    6: 'RUNNING',
    7: 'SHUFFLING',
    8: 'PICKING',
    9: 'JUMPING',
    10: 'CYCLING'
}

# ==========================================
# 2. MODEL DEFINITIONS
# ==========================================
class CNN1D(nn.Module):
    def __init__(self, num_classes=11):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(6, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(128 * 32, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (N,128,6) → (N,6,128)
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, num_classes=11):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: (N, 128, 6)
        lstm_out, (hidden, cell) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # (N, 128)
        out = self.fc(last_hidden)
        return out

class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=11):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(6, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # x: (N, 128, 6)
        x = x.permute(0, 2, 1)  # (N,6,128)
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.permute(0, 2, 1)
        lstm_out, (hidden, cell) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # (N, 128)
        out = self.fc(last_hidden)
        return out

# ==========================================
# 3. EMBEDDING EXTRACTOR (각 모델별로 다름)
# ==========================================
def extract_embeddings(model, model_type, X_tensor, device):
    """
    model: loaded PyTorch model
    model_type: 'cnn', 'lstm', 'cnn_lstm'
    X_tensor: torch.FloatTensor on correct device
    Returns: embeddings (numpy), predictions (numpy)
    """
    model.eval()
    embeddings = []
    predictions = []
    
    with torch.no_grad():
        batch_size = 256
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            
            if model_type == 'cnn':
                # CNN1D: Conv → Pool → Conv → Pool → Flatten → fc1 (임베딩)
                x = batch.permute(0, 2, 1)
                x = model.pool1(torch.relu(model.conv1(x)))
                x = model.pool2(torch.relu(model.conv2(x)))
                x = x.view(x.size(0), -1)
                emb = model.fc1(x)  # ← Embedding (N, 256)
                out = model.fc2(model.dropout(emb))
                
            elif model_type == 'lstm':
                # LSTM: LSTM 통과 → 마지막 hidden (임베딩)
                lstm_out, (hidden, cell) = model.lstm(batch)
                emb = lstm_out[:, -1, :]  # ← Embedding (N, 128)
                out = model.fc(emb)
                
            elif model_type == 'cnn_lstm':
                # CNN-LSTM: Conv → Pool 두 번 → LSTM → 마지막 hidden (임베딩)
                x = batch.permute(0, 2, 1)
                x = model.pool1(torch.relu(model.conv1(x)))
                x = model.pool2(torch.relu(model.conv2(x)))
                x = x.permute(0, 2, 1)  # (N, 32, 128)
                lstm_out, (hidden, cell) = model.lstm(x)
                emb = lstm_out[:, -1, :]  # ← Embedding (N, 128)
                out = model.fc(emb)
            
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            
            pred = torch.argmax(out, dim=1)
            embeddings.append(emb.cpu().numpy())
            predictions.append(pred.cpu().numpy())
    
    return np.vstack(embeddings), np.concatenate(predictions)

# ==========================================
# 4. LOAD MODEL
# ==========================================
model_map = {'cnn': CNN1D, 'lstm': LSTMClassifier, 'cnn_lstm': CNN_LSTM}
ModelClass = model_map[MODEL_TYPE]

# Checkpoint에서 num_classes 추론
checkpoint = torch.load(MODEL_PATH, map_location=device)
num_classes = None
for key in ['fc.weight', 'fc2.weight']:
    if key in checkpoint:
        num_classes = checkpoint[key].shape[0]
        break
if num_classes is None:
    num_classes = 11

print(f"Detected num_classes: {num_classes}")

model = ModelClass(num_classes=num_classes).to(device)
model.load_state_dict(checkpoint)
model.eval()
print("Model loaded successfully.")

# ==========================================
# 5. LOAD DATA
# ==========================================
X = np.load(f'{BASE_DIR}/X_train.npy') 
y = np.load(f'{BASE_DIR}/y_train.npy') - 1  # 0-based

meta = pd.read_csv(f'{META_DIR}/meta_train.csv')
dataset_names = meta['dataset_name'].values

# Sampling
np.random.seed(42)
idx = np.random.choice(len(X), size=min(SAMPLE_SIZE, len(X)), replace=False)

X_sample = X[idx]
y_sample = y[idx]
ds_sample = dataset_names[idx]

print(f"Data loaded: {X_sample.shape}")

# Convert to tensor (extract_embeddings 안에서 permute 처리)
X_t = torch.FloatTensor(X_sample).to(device)

# ==========================================
# 6. EXTRACT
# ==========================================
print("\nExtracting embeddings...")
embeddings, predictions = extract_embeddings(model, MODEL_TYPE, X_t, device)

print(f"Embeddings shape: {embeddings.shape}")  # CNN:(N,256), LSTM/CNN-LSTM:(N,128)
print(f"Predictions shape: {predictions.shape}")

# ==========================================
# 7. DIMENSIONALITY REDUCTION
# ==========================================
print("\nRunning PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(embeddings)
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

print("\nRunning t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(embeddings)

# ==========================================
# 8. PLOT FUNCTION (with legend)
# ==========================================
def plot_2d(X_2d, color_vals, title, label_map=None, cmap='tab20', save_path=None):
    """
    X_2d: (N, 2) 좌표
    color_vals: (N,) 클래스 번호 (0-based)
    label_map: {0: 'WALKING', 1: 'WALKING_UPSTAIRS', ...}
    """
    plt.figure(figsize=(13, 9))
    
    unique_labels = sorted(np.unique(color_vals))
    n_classes = len(unique_labels)
    
    # 클래스 개수만큼 색상 분할 (11개면 11개 색)
    colors = plt.cm.get_cmap(cmap, n_classes)
    
    for i, lbl in enumerate(unique_labels):
        mask = (color_vals == lbl)
        name = label_map[lbl] if (label_map and lbl in label_map) else f"Class {lbl}"
        
        plt.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=[colors(i)],      # 각 클래스별 고유 색
            label=name,
            alpha=0.6,
            s=10,
            edgecolors='none'
        )
    
    # 범례를 그래프 오른쪽에 배치
    plt.legend(
        title="Activity",
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0.,
        fontsize=9
    )
    
    plt.title(title, fontsize=14)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # 오른쪽에 legend 공간 확보
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

# ==========================================
# 9. PLOTS
# ==========================================
plot_2d(
    X_tsne, y_sample,
    title=f'{MODEL_TYPE.upper()} Embedding (t-SNE) - TRUE LABELS',
    label_map=LABEL_NAMES,
    save_path=f'figures/{MODEL_TYPE}_tsne_true_label.png'
)

plot_2d(
    X_tsne, predictions,
    title=f'{MODEL_TYPE.upper()} Embedding (t-SNE) - PREDICTIONS',
    label_map=LABEL_NAMES,
    save_path=f'figures/{MODEL_TYPE}_tsne_pred.png'
)

ds_map = {name: i for i, name in enumerate(np.unique(ds_sample))}
ds_numeric = np.array([ds_map[name] for name in ds_sample])
plot_2d(
    X_tsne, ds_numeric,
    title=f'{MODEL_TYPE.upper()} Embedding (t-SNE) - DATASET SOURCE',
    cmap='Set1',
    save_path=f'figures/{MODEL_TYPE}_tsne_dataset.png'
)

plot_2d(
    X_pca, y_sample,
    title=f'{MODEL_TYPE.upper()} Embedding (PCA) - TRUE LABELS',
    label_map=LABEL_NAMES,
    save_path=f'figures/{MODEL_TYPE}_pca_true_label.png'
)

print("\nDone")