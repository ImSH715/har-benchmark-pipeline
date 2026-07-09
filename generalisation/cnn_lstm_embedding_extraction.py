"""
Extract CNN-LSTM embeddings (before final FC layer) for Train/Test/Val.
Outputs: embeddings/ folder with .npy files
"""
import torch
import torch.nn as nn
import numpy as np
import os

print("=" * 60)
print("Extracting CNN-LSTM Embeddings")
print("=" * 60)

# ==========================================
# CONFIG
# ==========================================
MODEL_PATH = '../saved_models/cnn_lstm_20260630_142946.pth'  # Your .pth file
DATA_DIR = '../Dataset/Preprocessed/for_dl'
OUTPUT_DIR = '../embeddings'
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ==========================================
# MODEL DEFINITION (must match training architecture)
# ==========================================
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
        x = x.permute(0, 2, 1)
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return lstm_out[:, -1, :]  # Return embedding instead of logits

# Load model
checkpoint = torch.load(MODEL_PATH, map_location=device)
num_classes = checkpoint['fc.weight'].shape[0]

model = CNN_LSTM(num_classes=num_classes).to(device)
model.load_state_dict(checkpoint)
model.eval()
print(f"Model loaded: {MODEL_PATH} (num_classes={num_classes})")

# ==========================================
# EXTRACTION FUNCTION
# ==========================================
def extract_split(split_name):
    X = np.load(f'{DATA_DIR}/X_{split_name}.npy')
    print(f"\nProcessing {split_name}: {X.shape}")
    
    X_t = torch.FloatTensor(X).to(device)
    embeddings = []
    
    with torch.no_grad():
        batch_size = 256
        for i in range(0, len(X_t), batch_size):
            batch = X_t[i:i+batch_size]
            emb = model(batch)  # (N, 128)
            embeddings.append(emb.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    out_path = f'{OUTPUT_DIR}/cnn_lstm_emb_{split_name}.npy'
    np.save(out_path, embeddings)
    print(f"Saved: {out_path} | Shape: {embeddings.shape}")
    return embeddings

# ==========================================
# EXTRACT ALL SPLITS
# ==========================================
for split in ['train', 'val', 'test']:
    extract_split(split)

print("\n" + "=" * 60)
print("All embeddings extracted successfully!")
print(f"Location: {OUTPUT_DIR}/")
print("=" * 60)