import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
from torch.optim.lr_scheduler import StepLR


# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


data = pd.read_csv("insurance_data_preprocessed.csv")
# 只保留风险评分作为目标变量
X = data.drop(['avg_claim_amount', 'total_claims_paid', 'annual_medical_cost', 'claims_count', 'risk_score', 'is_high_risk'], axis=1, errors='ignore')
y = data['risk_score']  # 只预测风险评分

# 检查原始risk_score的范围
print(f"Original risk_score range: [{y.min():.3f}, {y.max():.3f}]")


# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42
)

# X_val, X_test, y_val, y_test = train_test_split(
#     X_test0, y_test, test_size=0.5,  
#     random_state=42
# )

print(f"Dataset Size:")
print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
# print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")    

# 转换为PyTorch张量（先创建CPU张量）
X_train_tensor = torch.FloatTensor(X_train.values)
# X_val_tensor = torch.FloatTensor(X_val.values)
X_test_tensor = torch.FloatTensor(X_test.values)

y_train_tensor = torch.FloatTensor(y_train.values)
# y_val_tensor = torch.FloatTensor(y_val.values)
y_test_tensor = torch.FloatTensor(y_test.values)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
batch_size = 64  # 使用更合理的批次大小
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

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
lr = 0.005
step_size, gamma = 100, 0.5
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size, gamma)


# 训练参数
num_epochs = 500
train_losses = []
val_losses = []
learning_rates = []

print("Start Training...")
time_start = time.time()    
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
    
    # # 将验证集数据移动到GPU
    # X_val_gpu = X_val_tensor.to(device)
    # y_val_gpu = y_val_tensor.to(device).view(-1, 1)
    
    # # 验证阶段
    # model.eval()
    # with torch.no_grad():
    #     val_pred_score = model(X_val_gpu)
    #     val_loss = criterion(val_pred_score, y_val_gpu)
    
    # # 学习率调度
    # scheduler.step(val_loss)
    # current_lr = optimizer.param_groups[0]['lr']
    # learning_rates.append(current_lr)
    
    # # 记录损失
    train_losses.append(epoch_train_loss / len(train_loader))
    # val_losses.append(val_loss.item())
    
    if (epoch + 1) % 10 == 0 :
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {train_losses[-1]:.6f}')
        # print(f'  Learning Rate: {current_lr:.6f}')
        
        # # 检查风险评分的统计信息
        # with torch.no_grad():
        #     print(f'  Risk Score - Min: {val_pred_score.min().item():.4f}, Max: {val_pred_score.max().item():.4f}, Mean: {val_pred_score.mean().item():.4f}')
        # print('-' * 50)
time_end = time.time()
print(f'Training Time: {time_end - time_start:.2f}s')





