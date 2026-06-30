"""
CNN-LSTM Hybrid for HAR Benchmark
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import json
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report

print("=" * 60)
print("CNN-LSTM Hybrid (PyTorch)")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# LOAD DATA
BASE_DIR = '../../Dataset/Preprocessed/for_dl'

X_train = np.load(f'{BASE_DIR}/X_train.npy')
X_val   = np.load(f'{BASE_DIR}/X_val.npy')
X_test  = np.load(f'{BASE_DIR}/X_test.npy')

y_train = np.load(f'{BASE_DIR}/y_train.npy') - 1
y_val   = np.load(f'{BASE_DIR}/y_val.npy') - 1
y_test  = np.load(f'{BASE_DIR}/y_test.npy') - 1

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# DATALOADER
X_train_t = torch.FloatTensor(X_train)
X_val_t   = torch.FloatTensor(X_val)
X_test_t  = torch.FloatTensor(X_test)

y_train_t = torch.LongTensor(y_train)
y_val_t   = torch.LongTensor(y_val)
y_test_t  = torch.LongTensor(y_test)

batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size)
test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size)

# MODEL DEFINITION
class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=11):
        super(CNN_LSTM, self).__init__()
        
        # ========== CNN Encoder ==========
        # Input: (batch, 6, 128)
        self.conv1 = nn.Conv1d(6, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)   # 128 → 64
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)   # 64 → 32
        
        
        self.lstm = nn.LSTM(
            input_size=128,      
            hidden_size=128,     # LSTM memory size
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # ========== Classifier ==========
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # → (batch, 6, 128)
        
        # CNN feature extraction
        x = self.pool1(self.relu1(self.conv1(x)))  # → (batch, 64, 64)
        x = self.pool2(self.relu2(self.conv2(x)))  # → (batch, 128, 32)
        
        # (batch, 128 channels, 32 length) → (batch, 32 seq_len, 128 features)
        x = x.permute(0, 2, 1)
        
        # LSTM temporal modeling
        # lstm_out: (batch, 32, 128), hidden: (2, batch, 128)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        last_hidden = lstm_out[:, -1, :]  # (batch, 128)
        
        # 분류
        out = self.fc(last_hidden)
        return out

model = CNN_LSTM(num_classes=11).to(device)
print(f"\nModel:\n{model}")

# LOSS / OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TRAINING LOOP
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
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()

model.load_state_dict(best_model_state)

# TEST EVALUATION
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
test_preds_original = np.array(test_preds) + 1
test_labels_original = np.array(test_labels) + 1

print(f"\nBest Val Accuracy: {best_val_acc:.4f}")
print(f"Test Accuracy:     {test_acc:.4f}")
print("\nClassification Report (Test):")
print(classification_report(test_labels_original, test_preds_original))

# SAVE
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

model_dir = '../../saved_models'
results_dir = '../../results'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

model_path = f'{model_dir}/cnn_lstm_{timestamp}.pth'
result_path = f'{results_dir}/cnn_lstm_{timestamp}.json'

torch.save(best_model_state, model_path)
print(f"\nModel saved: {model_path}")

results = {
    'model': 'CNN_LSTM',
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