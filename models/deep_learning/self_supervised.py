"""
SimCLR Self-Supervised Learning for Wearable HAR
Learns representations WITHOUT labels via contrastive learning.
Output: Saved encoder weights + extracted embeddings for K-means evaluation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from datetime import datetime

print("=" * 70)
print("SimCLR Self-Supervised Training")
print("=" * 70)

# ==========================================
# 0. CONFIGURATION
# ==========================================
DATA_DIR = '../../Dataset/Preprocessed/for_dl'
OUTPUT_DIR = '../../embeddings'
MODEL_DIR = '../../saved_models'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 512
EPOCHS = 10000
LR = 0.001
TEMPERATURE = 0.1
EMB_DIM = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ==========================================
# 1. DATASET WITH AUGMENTATION
# ==========================================
class IMUDataset(Dataset):
    def __init__(self, X):
        """
        X: np.array of shape (N, 128, 6)
        """
        self.X = torch.FloatTensor(X)
    
    def __len__(self):
        return len(self.X)
    
    def augment(self, x):
        """
        Creates two augmented views of the same window.
        x: (128, 6)
        """
        x1 = x.clone()
        x2 = x.clone()
        
        # Gaussian noise
        x1 = x1 + torch.randn_like(x1) * 0.05
        x2 = x2 + torch.randn_like(x2) * 0.05
        
        # Time shift (circular)
        if np.random.rand() > 0.5:
            shift = np.random.randint(5, 20)
            x2 = torch.roll(x2, shifts=shift, dims=0)
        
        # Amplitude scaling
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            x2 = x2 * scale
        
        return x1, x2
    
    def __getitem__(self, idx):
        x = self.X[idx]
        x1, x2 = self.augment(x)
        return x1, x2

# ==========================================
# 2. ENCODER + PROJECTION HEAD
# ==========================================
class SimCLR(nn.Module):
    def __init__(self, emb_dim=128):
        super(SimCLR, self).__init__()
        
        # Encoder (same architecture as supervised CNN-LSTM)
        self.conv1 = nn.Conv1d(6, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True, dropout=0.3)
        
        # Projection head (only used during SimCLR training)
        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim)
        )
    
    def forward(self, x):
        """
        x: (batch, 128, 6)
        Returns:
            h: embedding for downstream tasks (batch, 128)
            z: projection for contrastive loss (batch, emb_dim)
        """
        x = x.permute(0, 2, 1)  # (batch, 6, 128)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.permute(0, 2, 1)  # (batch, 32, 128)
        lstm_out, _ = self.lstm(x)
        h = lstm_out[:, -1, :]   # (batch, 128)
        z = self.projection(h)   # (batch, emb_dim)
        return h, z
    
    def get_embedding(self, x):
        """After training, use this to extract h without projection head."""
        with torch.no_grad():
            h, _ = self.forward(x)
        return h

# ==========================================
# 3. NT-Xent LOSS
# ==========================================
def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    Normalized Temperature-scaled Cross Entropy Loss.
    z_i, z_j: (batch, emb_dim) from two augmented views.
    """
    batch_size = z_i.size(0)
    
    # L2 normalize
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    
    # Concatenate: (2*batch, emb_dim)
    z = torch.cat([z_i, z_j], dim=0)
    
    # Similarity matrix: (2N, 2N)
    sim_matrix = torch.mm(z, z.t()) / temperature
    
    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)
    
    # Positive pairs: (i, i+N) and (i+N, i)
    pos_sim = torch.cat([
        torch.diag(sim_matrix, batch_size),   # sim(i, i+N)
        torch.diag(sim_matrix, -batch_size)   # sim(i+N, i)
    ], dim=0).unsqueeze(1)  # (2N, 1)
    
    # Logits: positives + all negatives
    logits = torch.cat([pos_sim, sim_matrix], dim=1)  # (2N, 2N+1)
    
    # Labels: positive is always at index 0
    labels = torch.zeros(2 * batch_size, device=z.device).long()
    
    return F.cross_entropy(logits, labels)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train(X_train):
    dataset = IMUDataset(X_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    model = SimCLR(emb_dim=EMB_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    print(f"\nStarting training for {EPOCHS} epochs...")
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for x1, x2 in loader:
            x1, x2 = x1.to(device), x2.to(device)
            
            # Forward pass for both views
            _, z1 = model(x1)
            _, z2 = model(x2)
            
            # Compute contrastive loss
            loss = nt_xent_loss(z1, z2, temperature=TEMPERATURE)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}]  Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'{MODEL_DIR}/simclr_best.pth')
    
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Model saved: {MODEL_DIR}/simclr_best.pth")
    return model

# ==========================================
# 5. EXTRACT EMBEDDINGS
# ==========================================
def extract_embeddings(model, X, split_name):
    model.eval()
    
    X_t = torch.FloatTensor(X).to(device)
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(X_t), 512):
            batch = X_t[i:i+512]
            emb = model.get_embedding(batch)
            embeddings.append(emb.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    save_path = f'{OUTPUT_DIR}/simclr_emb_{split_name}.npy'
    np.save(save_path, embeddings)
    print(f"Embeddings saved: {save_path} | Shape: {embeddings.shape}")
    return embeddings

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Load data
    print("\nLoading data...")
    X_train = np.load(f'{DATA_DIR}/X_train.npy')
    X_val = np.load(f'{DATA_DIR}/X_val.npy')
    X_test = np.load(f'{DATA_DIR}/X_test.npy')
    
    print(f"Train: {X_train.shape}")
    print(f"Val:   {X_val.shape}")
    print(f"Test:  {X_test.shape}")
    
    # Train SimCLR
    model = train(X_train)
    
    # Load best checkpoint
    print("\nLoading best checkpoint for embedding extraction...")
    model.load_state_dict(torch.load(f'{MODEL_DIR}/simclr_best.pth'))
    
    # Extract and save embeddings
    print("\nExtracting embeddings...")
    extract_embeddings(model, X_train, 'train')
    extract_embeddings(model, X_val, 'val')
    extract_embeddings(model, X_test, 'test')
    
    print("\n" + "=" * 70)
    print("DONE")
    print("Next: Run kmeans_on_embeddings.py with:")
    print("  EMB_DIR = '../embeddings'")
    print("  and use 'simclr_emb_train.npy' / 'simclr_emb_test.npy'")
    print("=" * 70)