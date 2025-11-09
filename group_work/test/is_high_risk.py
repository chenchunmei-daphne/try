import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

data = pd.read_csv("try/group_work/insurance_data_preprocessed.csv")

# 只保留高风险标志作为目标变量
X = data.drop(['avg_claim_amount', 'total_claims_paid', 'annual_medical_cost', 'claims_count', 'risk_score', 'is_high_risk'], axis=1, errors='ignore')
y = data['is_high_risk']  # 只预测是否为高风险

# 检查类别分布
print(f"Class distribution:")
print(y.value_counts())
print(f"High risk proportion: {y.mean():.3f}")

# 数据标准化
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
print("Features standardized with StandardScaler")

# 划分数据集 - 使用分层抽样
X_train, X_test0, y_train, y_test0 = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=y  # 保持类别比例
)

X_val, X_test, y_val, y_test = train_test_split(
    X_test0, y_test0, test_size=0.5,  
    random_state=42,
    stratify=y_test0
)

print(f"Dataset Size:")
print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")    

print(f"Training set class distribution: {y_train.value_counts().to_dict()}")
print(f"Validation set class distribution: {y_val.value_counts().to_dict()}")
print(f"Test set class distribution: {y_test.value_counts().to_dict()}")

# 转换为PyTorch张量（先创建CPU张量）
X_train_tensor = torch.FloatTensor(X_train.values)
X_val_tensor = torch.FloatTensor(X_val.values)
X_test_tensor = torch.FloatTensor(X_test.values)

y_train_tensor = torch.LongTensor(y_train.values)  # 使用LongTensor用于分类
y_val_tensor = torch.LongTensor(y_val.values)
y_test_tensor = torch.LongTensor(y_test.values)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

print(f"Batch size: {batch_size}, Total batches per epoch: {len(train_loader)}")

class HighRiskClassifier(nn.Module):
    def __init__(self, input_size, hidden_layers=[256, 128, 64], dropout_rate=0.3):
        super(HighRiskClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # 构建隐藏层
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # 输出层 - 二分类输出
        self.output_layer = nn.Sequential(
            nn.Linear(prev_size, 1),
            nn.Sigmoid()  # 输出概率
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.hidden_layers(x)
        probability = self.output_layer(features)
        return probability

# 初始化模型并移动到GPU
input_size = X_train_tensor.shape[1]
model = HighRiskClassifier(input_size=input_size).to(device)

# 定义损失函数和优化器 - 使用BCEWithLogitsLoss
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

# 训练参数
num_epochs = 100
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
learning_rates = []

print("Start Training...")
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    epoch_train_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch_X, batch_y in train_loader:
        # 将批次数据移动到GPU
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device).float().view(-1, 1)  # 转换为float用于BCE损失
        
        # 前向传播
        pred_prob = model(batch_X)
        
        # 计算损失
        loss = criterion(pred_prob, batch_y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_train_loss += loss.item()
        
        # 计算训练准确率
        pred_labels = (pred_prob > 0.5).float()
        train_correct += (pred_labels == batch_y).sum().item()
        train_total += batch_y.size(0)
    
    # 将验证集数据移动到GPU
    X_val_gpu = X_val_tensor.to(device)
    y_val_gpu = y_val_tensor.to(device).float().view(-1, 1)
    
    # 验证阶段
    model.eval()
    with torch.no_grad():
        val_pred_prob = model(X_val_gpu)
        val_loss = criterion(val_pred_prob, y_val_gpu)
        
        # 计算验证准确率
        val_pred_labels = (val_pred_prob > 0.5).float()
        val_accuracy = (val_pred_labels == y_val_gpu).float().mean()
    
    # 学习率调度
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    
    # 记录损失和准确率
    train_losses.append(epoch_train_loss / len(train_loader))
    val_losses.append(val_loss.item())
    train_accuracies.append(train_correct / train_total)
    val_accuracies.append(val_accuracy.item())
    
    if (epoch + 1) % 10 == 0 or epoch < 5:
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}')
        print(f'  Train Accuracy: {train_accuracies[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}')
        print(f'  Learning Rate: {current_lr:.6f}')
        
        # 检查预测概率的统计信息
        with torch.no_grad():
            print(f'  Predicted Probability - Min: {val_pred_prob.min().item():.4f}, Max: {val_pred_prob.max().item():.4f}, Mean: {val_pred_prob.mean().item():.4f}')
        print('-' * 50)

print("Training Completed!")

# 将测试集数据移动到GPU
X_test_gpu = X_test_tensor.to(device)

# 最终评估（使用测试集）
model.eval()
with torch.no_grad():
    test_pred_prob = model(X_test_gpu)
    
    # 移动到CPU进行后续处理
    test_pred_prob_np = test_pred_prob.cpu().numpy().flatten()
    test_pred_labels_np = (test_pred_prob_np > 0.5).astype(int)
    y_test_np = y_test_tensor.numpy()

# 计算最终指标
final_accuracy = accuracy_score(y_test_np, test_pred_labels_np)
final_auc = roc_auc_score(y_test_np, test_pred_prob_np)
final_precision = precision_score(y_test_np, test_pred_labels_np)
final_recall = recall_score(y_test_np, test_pred_labels_np)
final_f1 = f1_score(y_test_np, test_pred_labels_np)

print("\nFinal Test Results:")
print(f"Accuracy: {final_accuracy:.6f}")
print(f"AUC: {final_auc:.6f}")
print(f"Precision: {final_precision:.6f}")
print(f"Recall: {final_recall:.6f}")
print(f"F1-Score: {final_f1:.6f}")

# 检查预测概率的最终输出范围
print(f"\nPredicted Probability Range: [{test_pred_prob_np.min():.6f}, {test_pred_prob_np.max():.6f}]")

# 绘制损失曲线和准确率曲线
plt.figure(figsize=(15, 10))

# 损失曲线
plt.subplot(2, 3, 1)
plt.plot(train_losses, label='Train Loss', color='blue', alpha=0.7)
plt.plot(val_losses, label='Validation Loss', color='red', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('BCE Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 准确率曲线
plt.subplot(2, 3, 2)
plt.plot(train_accuracies, label='Train Accuracy', color='blue', alpha=0.7)
plt.plot(val_accuracies, label='Validation Accuracy', color='red', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# 学习率变化
plt.subplot(2, 3, 3)
plt.plot(learning_rates, color='purple', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid(True, alpha=0.3)

# ROC曲线
plt.subplot(2, 3, 4)
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test_np, test_pred_prob_np)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {final_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5, label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)

# 精确率-召回率曲线
plt.subplot(2, 3, 5)
from sklearn.metrics import precision_recall_curve
precision, recall, _ = precision_recall_curve(y_test_np, test_pred_prob_np)
plt.plot(recall, precision, color='green', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True, alpha=0.3)

# 混淆矩阵热图
plt.subplot(2, 3, 6)
cm = confusion_matrix(y_test_np, test_pred_labels_np)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Low Risk', 'Predicted High Risk'],
            yticklabels=['True Low Risk', 'True High Risk'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.show()

# 输出详细统计信息
print("\nDetailed Prediction Statistics:")
print(f"Predicted High Risk Proportion: {test_pred_labels_np.mean():.3f}")
print(f"True High Risk Proportion: {y_test_np.mean():.3f}")

# 混淆矩阵详细分析
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix Analysis:")
print(f"True Positives (TP): {tp}")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")

# 性能评估
print(f"\nPerformance Evaluation:")
if final_accuracy > 0.9:
    print(f"  Accuracy: {final_accuracy:.4f} - Excellent classification performance")
elif final_accuracy > 0.8:
    print(f"  Accuracy: {final_accuracy:.4f} - Good classification performance")
elif final_accuracy > 0.7:
    print(f"  Accuracy: {final_accuracy:.4f} - Fair classification performance")
else:
    print(f"  Accuracy: {final_accuracy:.4f} - Poor classification performance")

if final_auc > 0.9:
    print(f"  AUC: {final_auc:.4f} - Outstanding discriminative power")
elif final_auc > 0.8:
    print(f"  AUC: {final_auc:.4f} - Excellent discriminative power")
elif final_auc > 0.7:
    print(f"  AUC: {final_auc:.4f} - Acceptable discriminative power")
else:
    print(f"  AUC: {final_auc:.4f} - Poor discriminative power")

if final_f1 > 0.8:
    print(f"  F1-Score: {final_f1:.4f} - Excellent balance between precision and recall")
elif final_f1 > 0.6:
    print(f"  F1-Score: {final_f1:.4f} - Good balance between precision and recall")
else:
    print(f"  F1-Score: {final_f1:.4f} - Needs improvement in precision-recall balance")

# 不同阈值下的性能
print(f"\nPerformance at Different Thresholds:")
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
for threshold in thresholds:
    pred_labels_thresh = (test_pred_prob_np > threshold).astype(int)
    acc = accuracy_score(y_test_np, pred_labels_thresh)
    prec = precision_score(y_test_np, pred_labels_thresh)
    rec = recall_score(y_test_np, pred_labels_thresh)
    f1 = f1_score(y_test_np, pred_labels_thresh)
    print(f"  Threshold {threshold}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")