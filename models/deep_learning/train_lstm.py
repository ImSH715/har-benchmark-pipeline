"""
LSTM Baseline for HAR Benchmark
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
print("LSTM Baseline (PyTorch)")
print("=" * 60)

# DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# LOAD DATA
BASE_DIR = '../../Dataset/Preprocessed/for_dl'

X_train = np.load(f'{BASE_DIR}/X_train.npy')   # (17867, 128, 6)
X_val   = np.load(f'{BASE_DIR}/X_val.npy')     # (3817, 128, 6)
X_test  = np.load(f'{BASE_DIR}/X_test.npy')    # (3827, 128, 6)

y_train = np.load(f'{BASE_DIR}/y_train.npy') - 1  # 0-based
y_val   = np.load(f'{BASE_DIR}/y_val.npy') - 1
y_test  = np.load(f'{BASE_DIR}/y_test.npy') - 1

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# DATALOADER
# LSTM, no need permute
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
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, num_classes=11, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        
        # LSTM Level
        self.lstm = nn.LSTM(
            input_size=input_size,      # 6 (AccX/Y/Z + GyrX/Y/Z)
            hidden_size=hidden_size,    # 128
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Classifier : changing hidden layer
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch, 128, 6)
        
        # lstm_out: (batch, 128, hidden_size)
        # hidden: (num_layers, batch, hidden_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        # loading last time step
        last_output = lstm_out[:, -1, :]
        
        # classification
        out = self.fc(last_output)
        return out

model = LSTMClassifier(input_size=6, hidden_size=128, num_layers=2, num_classes=11).to(device)
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

model_path = f'{model_dir}/lstm_{timestamp}.pth'
result_path = f'{results_dir}/lstm_{timestamp}.json'

torch.save(best_model_state, model_path)
print(f"\nModel saved: {model_path}")

results = {
    'model': 'LSTM',
    'timestamp': timestamp,
    'num_epochs': num_epochs,
    'batch_size': batch_size,
    'learning_rate': 0.001,
    'hidden_size': 128,
    'num_layers': 2,
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