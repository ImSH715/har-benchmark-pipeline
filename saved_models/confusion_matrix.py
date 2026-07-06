"""
Universal Confusion Matrix Plotter
Works with: RandomForest (.pkl), XGBoost (.pkl), CNN (.pth), LSTM (.pth), CNN-LSTM (.pth)
"""
import sys
import os

def get_user_input():
    print("=" * 60)
    print("Universal Confusion Matrix Plotter")
    print("=" * 60)
    
    # 모델 파일 경로 입력받기
    model_path = input("\nEnter model file path (e.g., models/saved_models/cnn_lstm_20260630_142946.pth): ").strip()
    
    if not os.path.exists(model_path):
        print(f"ERROR: File not found: {model_path}")
        sys.exit(1)
    
    # 모델 타입 자동 감지
    filename = os.path.basename(model_path).lower()
    if 'rf_' in filename or 'random' in filename:
        model_type = 'sklearn'
    elif 'xgb_' in filename or 'xgboost' in filename:
        model_type = 'sklearn'
    elif 'svm_' in filename:
        model_type = 'sklearn'
    elif 'cnn_lstm' in filename or 'cnn-lstm' in filename:
        model_type = 'cnn_lstm'
    elif 'lstm_' in filename:
        model_type = 'lstm'
    elif 'cnn_' in filename:
        model_type = 'cnn'
    else:
        print("\nCould not auto-detect model type.")
        print("Options: 1=sklearn(RF/XGB/SVM), 2=cnn, 3=lstm, 4=cnn_lstm")
        choice = input("Select model type (1-4): ").strip()
        type_map = {'1': 'sklearn', '2': 'cnn', '3': 'lstm', '4': 'cnn_lstm'}
        model_type = type_map.get(choice, 'cnn_lstm')
    
    # 데이터 경로 (기본값 제공)
    default_data = 'Dataset/Preprocessed/for_ml' if model_type == 'sklearn' else 'Dataset/Preprocessed/for_dl'
    data_dir = input(f"Enter data directory [{default_data}]: ").strip()
    if not data_dir:
        data_dir = default_data
    
    # 출력 파일명
    default_out = f"figures/cm_{os.path.splitext(filename)[0]}.png"
    out_path = input(f"Output image path [{default_out}]: ").strip()
    if not out_path:
        out_path = default_out
    
    return model_path, model_type, data_dir, out_path

# ==========================================
# MAIN
# ==========================================
model_path, model_type, data_dir, out_path = get_user_input()

print(f"\nModel: {model_path}")
print(f"Type:  {model_type}")
print(f"Data:  {data_dir}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# LOAD DATA
if model_type == 'sklearn':
    X_test = pd.read_csv(f'{data_dir}/X_test_features.csv').values
else:
    X_test = np.load(f'{data_dir}/X_test.npy')

y_test = np.load(f'{data_dir}/y_test.npy')
print(f"Test data: {X_test.shape}, Labels: {sorted(np.unique(y_test))}")

# LOAD MODEL & PREDICT
if model_type == 'sklearn':
    import pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    y_pred = model.predict(X_test)
    
else:
    import torch
    import torch.nn as nn
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model
    class CNN1D(nn.Module):
        def __init__(self, num_classes=11):
            super().__init__()
            self.conv1 = nn.Conv1d(6, 64, 3, padding=1)
            self.pool1 = nn.MaxPool1d(2)
            self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
            self.pool2 = nn.MaxPool1d(2)
            self.fc1 = nn.Linear(128*32, 256)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(256, num_classes)
        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = self.pool1(torch.relu(self.conv1(x)))
            x = self.pool2(torch.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.dropout(x)
            return self.fc2(x)
    
    class LSTMClassifier(nn.Module):
        def __init__(self, input_size=6, hidden_size=128, num_layers=2, num_classes=11):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
            self.fc = nn.Linear(hidden_size, num_classes)
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :])
    
    class CNN_LSTM(nn.Module):
        def __init__(self, num_classes=11):
            super().__init__()
            self.conv1 = nn.Conv1d(6, 64, 3, padding=1)
            self.pool1 = nn.MaxPool1d(2)
            self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
            self.pool2 = nn.MaxPool1d(2)
            self.lstm = nn.LSTM(128, 128, 2, batch_first=True, dropout=0.3)
            self.fc = nn.Linear(128, num_classes)
        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = self.pool1(torch.relu(self.conv1(x)))
            x = self.pool2(torch.relu(self.conv2(x)))
            x = x.permute(0, 2, 1)
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :])
    
    model_map = {'cnn': CNN1D, 'lstm': LSTMClassifier, 'cnn_lstm': CNN_LSTM}
    ModelClass = model_map[model_type]
    
    checkpoint = torch.load(model_path, map_location=device)
    
    num_classes = None
    for key in ['fc.weight', 'fc2.weight']:
        if key in checkpoint:
            num_classes = checkpoint[key].shape[0]
            break
    
    if num_classes is None:
        print("WARNING: Could not infer num_classes from checkpoint. Defaulting to 11.")
        num_classes = 11
    
    print(f"Checkpoint num_classes: {num_classes}")
    
    model = ModelClass(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    X_t = torch.FloatTensor(X_test)
    if model_type in ['cnn', 'cnn_lstm']:
        X_t = X_t.permute(0, 2, 1) if model_type == 'cnn' else X_t
    
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_t), 256):
            batch = X_t[i:i+256].to(device)
            outputs = model(batch)
            _, pred = torch.max(outputs, 1)
            all_preds.extend(pred.cpu().numpy())
    
    y_pred = np.array(all_preds) + 1  # 0-based → 1-based

# ==========================================
# 3. COMPUTE METRICS
# ==========================================
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.4f}")

# 라벨: 체크포인트 기준 전체 클래스 (1~11)를 CM에 포함
# 데이터에 class 8이 없어도 '빈 행'으로 표시되어 모델이 8을 예측하는지 확인 가능
labels = list(range(1, num_classes + 1))
label_names = {
    1: 'Walk', 2: 'Upstairs', 3: 'Downstairs', 4: 'Sit',
    5: 'Stand', 6: 'Lay', 7: 'Run', 8: 'Shuffle',
    9: 'Pick', 10: 'Jump', 11: 'Cycling'
}
tick_labels = [label_names.get(l, str(l)) for l in labels]

cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# ==========================================
# 4. PLOT
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Absolute count
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=tick_labels, yticklabels=tick_labels)
axes[0].set_title(f'Absolute Count\nAccuracy: {acc:.2%}', fontsize=12)
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# Normalized
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
            xticklabels=tick_labels, yticklabels=tick_labels)
axes[1].set_title('Normalized (Row %)', fontsize=12)
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.suptitle(f'Confusion Matrix: {os.path.basename(model_path)}', fontsize=14)
plt.tight_layout()

os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"\nSaved: {out_path}")
plt.show()

# ==========================================
# 5. PRINT TOP MISCLASSIFICATIONS
# ==========================================
print("\n" + "=" * 60)
print("Top Misclassifications")
print("=" * 60)
misclass = []
for i, true_l in enumerate(labels):
    for j, pred_l in enumerate(labels):
        if i != j and cm[i, j] > 0:
            misclass.append({
                'True': label_names.get(true_l, true_l),
                'Pred': label_names.get(pred_l, pred_l),
                'Count': cm[i, j],
                'Pct': f"{cm_norm[i,j]*100:.1f}%"
            })

misclass_df = pd.DataFrame(misclass).sort_values('Count', ascending=False)
print(misclass_df.head(10).to_string(index=False))
print("=" * 60)