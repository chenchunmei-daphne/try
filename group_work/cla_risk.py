import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

data = pd.read_csv("try/group_work/insurance_data_preprocessed.csv")

X = data.drop(['avg_claim_amount', 'total_claims_paid', 'annual_medical_cost', 'claims_count', 'risk_score', 'is_high_risk'], axis=1, errors='ignore')
y = data.get(['risk_score', 'is_high_risk'])

# 检查原始risk_score的范围，如果需要可以手动归一化
print(f"Original risk_score range: [{y['risk_score'].min():.3f}, {y['risk_score'].max():.3f}]")

# 如果risk_score不在0~1范围内，进行归一化
if y['risk_score'].max() > 1.0 or y['risk_score'].min() < 0.0:
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    y['risk_score'] = scaler.fit_transform(y[['risk_score']])
    print(f"Normalized risk_score range: [{y['risk_score'].min():.3f}, {y['risk_score'].max():.3f}]")

# 划分数据集
X_train, X_test0, y_train, y_test0 = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=y['is_high_risk']
)

X_val, X_test, y_val, y_test = train_test_split(
    X_test0, y_test0, test_size=0.5,  
    random_state=42, stratify=y_test0['is_high_risk'])

print(f"Dataset Size:")
print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")    

# 转换为PyTorch张量并移动到GPU
X_train_tensor = torch.FloatTensor(X_train.values).to(device)
X_val_tensor = torch.FloatTensor(X_val.values).to(device)
X_test_tensor = torch.FloatTensor(X_test.values).to(device)

y_train_tensor = torch.FloatTensor(y_train.values).to(device)
y_val_tensor = torch.FloatTensor(y_val.values).to(device)
y_test_tensor = torch.FloatTensor(y_test.values).to(device)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
batch_size = int(len(X_train_tensor) / 10)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

class MultiTaskRiskNet(nn.Module):
    def __init__(self, input_size, share_hidden=[128, 64], regress_hidden=[32, 1], class_hidden=[32, 1], dropout_rate=0.3):
        super(MultiTaskRiskNet, self).__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, share_hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(share_hidden[0], share_hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 回归分支：预测风险评分 (0~1)
        self.regression_branch = nn.Sequential(
            nn.Linear(share_hidden[1], regress_hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(regress_hidden[0], regress_hidden[1]),
            # 使用Sigmoid确保输出在0~1之间
            nn.Sigmoid()  
        )
        
        self.classification_branch = nn.Sequential(
            nn.Linear(share_hidden[1] + 1, class_hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(class_hidden[0], class_hidden[1]),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        shared_features = self.shared_layers(x)
        risk_score = self.regression_branch(shared_features)
        
        # 确保risk_score在0~1之间（Sigmoid已经保证，这里再加一个clamp确保）
        risk_score = torch.clamp(risk_score, 1e-6, 1-1e-6)  # 避免0和1的极端值
        
        features = torch.cat([shared_features, risk_score], dim=-1)
        risk_class_prob = self.classification_branch(features)
        return risk_score, risk_class_prob

# 初始化模型并移动到GPU
input_size = X_train_tensor.shape[1]
model = MultiTaskRiskNet(input_size=input_size).to(device)

# 定义损失函数和优化器
lr = 0.01
w = 50 ## 权重
lr_step_size = 50
regression_criterion = nn.MSELoss()
classification_criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
lr_scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=0.5)

# 训练参数
num_epochs = 50
total_rain_losses = []
total_val_losses = []
train_accuracies = []
val_accuracies = []
regression_losses = []
classification_losses = []

print("Start Training...")
time_start = time.time()
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    epoch_reg_loss = 0
    epoch_cls_loss = 0
    epoch_total_loss = 0
    train_correct = 0
    train_total = 0
    
    # for batch_X, batch_y in train_loader:
    for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
        batch_y_score = batch_y[:, 0].view(-1, 1)  # 风险评分
        batch_y_class = batch_y[:, 1].view(-1, 1)  # 风险等级
        
        # 前向传播
        pred_score, pred_class = model(batch_X)
        
        # 计算损失
        loss_reg = regression_criterion(pred_score, batch_y_score)
        loss_cls = classification_criterion(pred_class, batch_y_class)
        total_loss = w * loss_reg + loss_cls
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        epoch_reg_loss += loss_reg.item()
        epoch_cls_loss += loss_cls.item()
        epoch_total_loss += total_loss.item()
        
        # 计算训练准确率
        pred_labels = (pred_class > 0.5).float()
        train_correct += (pred_labels == batch_y_class).sum().item()
        train_total += batch_y_class.size(0)

    lr_scheduler.step()

    # 验证阶段（使用验证集）
    model.eval()
    with torch.no_grad():
        val_pred_score, val_pred_class = model(X_val_tensor)
        val_y_score = y_val_tensor[:, 0].view(-1, 1)
        val_y_class = y_val_tensor[:, 1].view(-1, 1)
        
        val_loss_reg = regression_criterion(val_pred_score, val_y_score)
        val_loss_cls = classification_criterion(val_pred_class, val_y_class)
        val_total_loss = val_loss_reg + val_loss_cls
        
        # 计算验证准确率
        val_pred_labels = (val_pred_class > 0.5).float()
        val_accuracy = (val_pred_labels == val_y_class).float().mean()
    
    # 记录损失和准确率
    total_rain_losses.append(epoch_total_loss / len(train_loader))
    total_val_losses.append(val_total_loss.item())
    regression_losses.append(epoch_reg_loss / len(train_loader))
    classification_losses.append(epoch_cls_loss / len(train_loader))
    train_accuracies.append(train_correct / train_total)
    val_accuracies.append(val_accuracy.item())
    
    if epoch  % 10 == 0  or epoch < 5:
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {total_rain_losses[-1]:.4f}, Val Loss: {total_val_losses[-1]:.4f}')
        print(f'  Regression Loss: {regression_losses[-1]:.4f}, Classification Loss: {classification_losses[-1]:.4f}')
        print(f'  Train Accuracy: {train_accuracies[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}')
        
        # 检查risk_score的输出范围
        with torch.no_grad():
            sample_output = val_pred_score
            print(f'  Risk Score Range: [{sample_output.min().item():.4f}, {sample_output.max().item():.4f}]')
        print('-' * 50)

time_end = time.time()
print(f"traning time: {time_end - time_start:.2f} s")

# 最终评估（使用测试集）
model.eval()
with torch.no_grad():
    test_pred_score, test_pred_class = model(X_test_tensor)
    
    # 移动到CPU进行后续处理
    test_pred_score_np = test_pred_score.cpu().numpy().flatten()
    test_pred_class_np = test_pred_class.cpu().numpy().flatten()
    test_pred_labels_np = (test_pred_class_np > 0.5).astype(int)
    
    y_test_score_np = y_test_tensor[:, 0].cpu().numpy()
    y_test_class_np = y_test_tensor[:, 1].cpu().numpy()

# 计算最终指标
final_rmse = np.sqrt(mean_squared_error(y_test_score_np, test_pred_score_np))
final_accuracy = accuracy_score(y_test_class_np, test_pred_labels_np)
final_auc = roc_auc_score(y_test_class_np, test_pred_class_np)

print("\nFinal Test Results:")
print(f"Regression Task - RMSE: {final_rmse:.4f}")
print(f"Classification Task - Accuracy: {final_accuracy:.4f}, AUC: {final_auc:.4f}")

# 检查risk_score的最终输出范围
print(f"\nRisk Score Output Range: [{test_pred_score_np.min():.6f}, {test_pred_score_np.max():.6f}]")

# 绘制损失曲线
plt.figure(figsize=(15, 10))

# 总损失曲线
plt.subplot(2, 3, 1)
total_rain_losses = np.log10(np.array(total_rain_losses))
total_val_losses = np.log10(np.array(total_val_losses))
plt.plot(total_rain_losses, label='Train Loss', color='blue', alpha=0.7)
plt.plot(total_val_losses, label='Validation Loss', color='red', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('log10(Loss)')
plt.title('Total Loss Curve')
plt.legend()
plt.grid(True, alpha=0.3)

# 回归和分类损失曲线
plt.subplot(2, 3, 2)
regression_losses = np.log10(np.array(regression_losses))
classification_losses = np.log10(np.array(classification_losses))
plt.plot(regression_losses, label='Regression Loss', color='green', alpha=0.7)
plt.plot(classification_losses, label='Classification Loss', color='orange', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('log10(Loss)')
plt.title('Regression vs Classification Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 准确率曲线
plt.subplot(2, 3, 3)
plt.plot(train_accuracies, label='Train Accuracy', color='blue', alpha=0.7)
plt.plot(val_accuracies, label='Validation Accuracy', color='red', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Classification Accuracy Curve')
plt.legend()
plt.grid(True, alpha=0.3)

# 风险评分分布
plt.subplot(2, 3, 4)
for class_val in [0, 1]:
    mask = y_test_class_np == class_val
    plt.hist(test_pred_score_np[mask], alpha=0.7, label=f'True Class {class_val}', bins=20)
plt.xlabel('Predicted Risk Score')
plt.ylabel('Frequency')
plt.title('Risk Score Distribution by True Class')
plt.legend()
plt.grid(True, alpha=0.3)

# 预测 vs 真实风险评分
plt.subplot(2, 3, 5)
plt.scatter(y_test_score_np, test_pred_score_np, alpha=0.6)
plt.plot([0, 1], [0, 1], 'r--', alpha=0.8)
plt.xlabel('True Risk Score')
plt.ylabel('Predicted Risk Score')
plt.title('Risk Score Prediction Performance')
plt.grid(True, alpha=0.3)

# ROC曲线
plt.subplot(2, 3, 6)
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test_class_np, test_pred_class_np)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {final_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 输出统计信息
print("\nPrediction Statistics:")
print(f"Predicted Risk Score Range: [{test_pred_score_np.min():.6f}, {test_pred_score_np.max():.6f}]")
print(f"High Risk Sample Proportion: {test_pred_labels_np.mean():.3f}")
print(f"True High Risk Sample Proportion: {y_test_class_np.mean():.3f}")

# 检查高风险和低风险组的评分差异
high_risk_mask = test_pred_labels_np == 1
if np.sum(high_risk_mask) > 0 and np.sum(~high_risk_mask) > 0:
    high_risk_avg_score = test_pred_score_np[high_risk_mask].mean()
    low_risk_avg_score = test_pred_score_np[~high_risk_mask].mean()
    print(f"Predicted High Risk Group Average Score: {high_risk_avg_score:.3f}")
    print(f"Predicted Low Risk Group Average Score: {low_risk_avg_score:.3f}")
    print(f"Score Difference: {high_risk_avg_score - low_risk_avg_score:.3f}")