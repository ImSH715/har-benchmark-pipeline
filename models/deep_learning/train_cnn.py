"""
1D CNN Baseline for HAR Benchmark
Input:  Raw time-series (N, 128, 6)
Output: Model weights (.pth) + evaluation results (.json)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report

print("=" * 60)
print("1D CNN Baseline (PyTorch)")
print("=" * 60)

# 1. DEVICE SETUP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. LOAD DATA
BASE_DIR = '../../Dataset/Preprocessed/for_dl'

X_train = np.load(f'{BASE_DIR}/X_train.npy')   # (17867, 128, 6)
X_val   = np.load(f'{BASE_DIR}/X_val.npy')     # (3817, 128, 6)
X_test  = np.load(f'{BASE_DIR}/X_test.npy')    # (3827, 128, 6)

y_train = np.load(f'{BASE_DIR}/y_train.npy')   # (17867,)
y_val   = np.load(f'{BASE_DIR}/y_val.npy')     # (3817,)
y_test  = np.load(f'{BASE_DIR}/y_test.npy')    # (3827,)

# 0-base conversion for CNN
y_train = y_train - 1
y_val   = y_val - 1
y_test  = y_test - 1

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ==========================================
# 3. DATASET / DATALOADER
# ==========================================
# PyTorch : tensor conversion
# Conv1d 
X_train_t = torch.FloatTensor(X_train).permute(0, 2, 1)
X_val_t   = torch.FloatTensor(X_val).permute(0, 2, 1)
X_test_t  = torch.FloatTensor(X_test).permute(0, 2, 1)

y_train_t = torch.LongTensor(y_train)
y_val_t   = torch.LongTensor(y_val)
y_test_t  = torch.LongTensor(y_test)

# DataLoader Creation
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size)
test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size)

# 4. MODEL DEFINITION
class CNN1D(nn.Module):
    def __init__(self, num_classes=11):
        super(CNN1D, self).__init__()
        
        # Block 1: (6, 128) → (64, 128) → (64, 64)
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Block 2: (64, 64) → (128, 64) → (128, 32)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Classifier
        # 128 channels * 32 length = 4096
        self.fc1 = nn.Linear(128 * 32, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN1D(num_classes=11).to(device)
print(f"\nModel:\n{model}")

# 5. LOSS / OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. TRAINING LOOP
num_epochs = 30
best_val_acc = 0.0

print("\nTraining...")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(batch_y.numpy())
    
    val_acc = accuracy_score(val_labels, val_preds)
    print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {train_loss/len(train_loader):.4f}  Val Acc: {val_acc:.4f}")
    
    # Best model save
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()

# Best weights load
model.load_state_dict(best_model_state)

# 7. TEST EVALUATION
print("\nEvaluating on Test set...")
model.eval()
test_preds = []
test_labels = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        _, predicted = torch.max(outputs, 1)
        
        test_preds.extend(predicted.cpu().numpy())
        test_labels.extend(batch_y.numpy())

test_acc = accuracy_score(test_labels, test_preds)

# Conversion back to prediction
test_preds_original = np.array(test_preds) + 1
test_labels_original = np.array(test_labels) + 1

print(f"\nBest Val Accuracy: {best_val_acc:.4f}")
print(f"Test Accuracy:     {test_acc:.4f}")
print("\nClassification Report (Test):")
print(classification_report(test_labels_original, test_preds_original))

# 8. SAVE MODEL & RESULTS
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

model_dir = '../../saved_models'
results_dir = '../../results'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

model_path = f'{model_dir}/cnn1d_{timestamp}.pth'
result_path = f'{results_dir}/cnn1d_{timestamp}.json'

# PyTorch state_dict()
torch.save(best_model_state, model_path)
print(f"\nModel saved: {model_path}")

results = {
    'model': 'CNN1D',
    'timestamp': timestamp,
    'num_epochs': num_epochs,
    'batch_size': batch_size,
    'learning_rate': 0.001,
    'best_val_accuracy': float(best_val_acc),
    'test_accuracy': float(test_acc),
    'classification_report': classification_report(test_labels_original, test_preds_original, output_dict=True)
}

with open(result_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved: {result_path}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)