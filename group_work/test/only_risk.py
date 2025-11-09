import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

data = pd.read_csv("try/group_work/insurance_data_preprocessed.csv")

# 只保留风险评分作为目标变量
X = data.drop(['avg_claim_amount', 'total_claims_paid', 'annual_medical_cost', 'claims_count', 'risk_score', 'is_high_risk'], axis=1, errors='ignore')
y = data['risk_score']  # 只预测风险评分

# 检查原始risk_score的范围
print(f"Original risk_score range: [{y.min():.3f}, {y.max():.3f}]")

# 数据标准化
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
print("Features standardized with StandardScaler")

# 划分数据集
X_train, X_test0, y_train, y_test0 = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_test0, y_test0, test_size=0.5,  
    random_state=42
)

print(f"Dataset Size:")
print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")    

# 转换为PyTorch张量（先创建CPU张量）
X_train_tensor = torch.FloatTensor(X_train.values)
X_val_tensor = torch.FloatTensor(X_val.values)
X_test_tensor = torch.FloatTensor(X_test.values)

y_train_tensor = torch.FloatTensor(y_train.values)
y_val_tensor = torch.FloatTensor(y_val.values)
y_test_tensor = torch.FloatTensor(y_test.values)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
batch_size = 64  # 使用更合理的批次大小
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

print(f"Batch size: {batch_size}, Total batches per epoch: {len(train_loader)}")

class RiskScoreNet(nn.Module):
    def __init__(self, input_size, hidden_layers=[256, 128, 64], dropout_rate=0.2):
        super(RiskScoreNet, self).__init__()
        
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
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(prev_size, 1),
            nn.Sigmoid()  # 确保输出在0~1之间
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
        risk_score = self.output_layer(features)
        return risk_score

# 初始化模型并移动到GPU
input_size = X_train_tensor.shape[1]
model = RiskScoreNet(input_size=input_size).to(device)

# 定义损失函数和优化器
lr = 0.001
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

# 训练参数
num_epochs = 100
train_losses = []
val_losses = []
learning_rates = []

print("Start Training...")
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    epoch_train_loss = 0
    
    for batch_X, batch_y in train_loader:
        # 将批次数据移动到GPU
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device).view(-1, 1)  # 确保形状正确
        
        # 前向传播
        pred_score = model(batch_X)
        
        # 计算损失
        loss = criterion(pred_score, batch_y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_train_loss += loss.item()
    
    # 将验证集数据移动到GPU
    X_val_gpu = X_val_tensor.to(device)
    y_val_gpu = y_val_tensor.to(device).view(-1, 1)
    
    # 验证阶段
    model.eval()
    with torch.no_grad():
        val_pred_score = model(X_val_gpu)
        val_loss = criterion(val_pred_score, y_val_gpu)
    
    # 学习率调度
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    
    # 记录损失
    train_losses.append(epoch_train_loss / len(train_loader))
    val_losses.append(val_loss.item())
    
    if (epoch + 1) % 10 == 0 or epoch < 5:
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}')
        print(f'  Learning Rate: {current_lr:.6f}')
        
        # 检查风险评分的统计信息
        with torch.no_grad():
            print(f'  Risk Score - Min: {val_pred_score.min().item():.4f}, Max: {val_pred_score.max().item():.4f}, Mean: {val_pred_score.mean().item():.4f}')
        print('-' * 50)

print("Training Completed!")

# 将测试集数据移动到GPU
X_test_gpu = X_test_tensor.to(device)

# 最终评估（使用测试集）
model.eval()
with torch.no_grad():
    test_pred_score = model(X_test_gpu)
    
    # 移动到CPU进行后续处理
    test_pred_score_np = test_pred_score.cpu().numpy().flatten()
    y_test_np = y_test_tensor.numpy()

# 计算最终指标
final_rmse = np.sqrt(mean_squared_error(y_test_np, test_pred_score_np))
final_mae = np.mean(np.abs(y_test_np - test_pred_score_np))
final_r2 = r2_score(y_test_np, test_pred_score_np)

print("\nFinal Test Results:")
print(f"RMSE: {final_rmse:.6f}")
print(f"MAE: {final_mae:.6f}")
print(f"R² Score: {final_r2:.6f}")

# 检查risk_score的最终输出范围
print(f"\nRisk Score Output Range: [{test_pred_score_np.min():.6f}, {test_pred_score_np.max():.6f}]")
print(f"True Risk Score Range: [{y_test_np.min():.6f}, {y_test_np.max():.6f}]")

# 绘制损失曲线
plt.figure(figsize=(15, 10))

# 损失曲线
plt.subplot(2, 3, 1)
plt.plot(train_losses, label='Train Loss', color='blue', alpha=0.7)
plt.plot(val_losses, label='Validation Loss', color='red', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Log10损失曲线
plt.subplot(2, 3, 2)
train_losses_log = np.log10(np.array(train_losses) + 1e-8)
val_losses_log = np.log10(np.array(val_losses) + 1e-8)
plt.plot(train_losses_log, label='Train Loss (log10)', color='blue', alpha=0.7)
plt.plot(val_losses_log, label='Validation Loss (log10)', color='red', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Log10(Loss)')
plt.title('Loss Curve (Log10 Scale)')
plt.legend()
plt.grid(True, alpha=0.3)

# 学习率变化
plt.subplot(2, 3, 3)
plt.plot(learning_rates, color='purple', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid(True, alpha=0.3)

# 预测 vs 真实风险评分
plt.subplot(2, 3, 4)
plt.scatter(y_test_np, test_pred_score_np, alpha=0.6)
plt.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Perfect Prediction')
plt.xlabel('True Risk Score')
plt.ylabel('Predicted Risk Score')
plt.title('Risk Score Prediction Performance')
plt.legend()
plt.grid(True, alpha=0.3)

# 残差图
plt.subplot(2, 3, 5)
residuals = test_pred_score_np - y_test_np
plt.scatter(test_pred_score_np, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
plt.xlabel('Predicted Risk Score')
plt.ylabel('Residuals (Predicted - True)')
plt.title('Prediction Residuals')
plt.grid(True, alpha=0.3)

# 预测值分布
plt.subplot(2, 3, 6)
plt.hist(test_pred_score_np, bins=50, alpha=0.7, color='blue', label='Predicted')
plt.hist(y_test_np, bins=50, alpha=0.7, color='red', label='True')
plt.xlabel('Risk Score')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted vs True Scores')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 输出详细统计信息
print("\nDetailed Prediction Statistics:")
print(f"Predicted Risk Score - Min: {test_pred_score_np.min():.6f}")
print(f"Predicted Risk Score - Max: {test_pred_score_np.max():.6f}")
print(f"Predicted Risk Score - Mean: {test_pred_score_np.mean():.6f}")
print(f"Predicted Risk Score - Std: {test_pred_score_np.std():.6f}")

print(f"\nTrue Risk Score - Min: {y_test_np.min():.6f}")
print(f"True Risk Score - Max: {y_test_np.max():.6f}")
print(f"True Risk Score - Mean: {y_test_np.mean():.6f}")
print(f"True Risk Score - Std: {y_test_np.std():.6f}")

# 分位数分析
print(f"\nQuantile Analysis:")
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
pred_quantiles = np.quantile(test_pred_score_np, quantiles)
true_quantiles = np.quantile(y_test_np, quantiles)

for q, pred_q, true_q in zip(quantiles, pred_quantiles, true_quantiles):
    print(f"  {q:.0%} Quantile - Predicted: {pred_q:.4f}, True: {true_q:.4f}, Diff: {pred_q-true_q:.4f}")

# 性能评估
print(f"\nPerformance Evaluation:")
if final_rmse < 0.05:
    print(f"  RMSE: {final_rmse:.4f} - Excellent prediction accuracy")
elif final_rmse < 0.1:
    print(f"  RMSE: {final_rmse:.4f} - Good prediction accuracy")
elif final_rmse < 0.15:
    print(f"  RMSE: {final_rmse:.4f} - Fair prediction accuracy")
else:
    print(f"  RMSE: {final_rmse:.4f} - Poor prediction accuracy")

if final_r2 > 0.8:
    print(f"  R²: {final_r2:.4f} - Excellent model fit")
elif final_r2 > 0.6:
    print(f"  R²: {final_r2:.4f} - Good model fit")
elif final_r2 > 0.4:
    print(f"  R²: {final_r2:.4f} - Fair model fit")
else:
    print(f"  R²: {final_r2:.4f} - Poor model fit")