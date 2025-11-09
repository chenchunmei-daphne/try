import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns

data = pd.read_csv("insurance_data_preprocessed.csv")



X = data.drop(['avg_claim_amount', 'total_claims_paid', 'annual_medical_cost', 'claims_count', 'risk_score', 'is_high_risk'], axis=1, errors='ignore')
y = data.get(['risk_score', 'is_high_risk'])
X.info()
# 划分训练集和测试集
X_train0, X_test0, y_train0, y_test0 = train_test_split(
    X, y, 
    test_size=0.2,  # 测试集比例
    random_state=42,  # 随机种子，确保可重复性
    stratify=y['is_high_risk']  # 按高风险标签分层抽样
)

# 转换为PyTorch张量
X_train = torch.FloatTensor(X_train0.values)
X_test = torch.FloatTensor(X_test0.values)
y_train = torch.FloatTensor(y_train0.values)
y_test = torch.FloatTensor(y_test0.values) 



# 创建数据集
train_dataset = TensorDataset(X_train, y_train)

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,  # 每个epoch打乱数据
    num_workers=1 # 多进程加载数据
)


class MultiTaskRiskNet(nn.Module):
    def __init__(self, input_size, share_hidden=[128, 64], regress_hidden=[32, 1], class_hidden=[32, 1], dropout_rate=0.3):
        super(MultiTaskRiskNet, self).__init__()
        
        # 共享的特征提取层
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
            nn.Sigmoid()  # 输出在0~1之间
        )
        
        # 分类分支：现在输入维度是 share_hidden[1] + 1（增加了风险评分特征）
        self.classification_branch = nn.Sequential(
            nn.Linear(share_hidden[1] + 1, class_hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(class_hidden[0], class_hidden[1]),
            nn.Sigmoid()  # 输出概率
        )
    
    def forward(self, x):
        # 共享特征提取
        shared_features = self.shared_layers(x)
    
        risk_score = self.regression_branch(shared_features)  # 连续评分
        
        # 将共享特征和预测的风险评分拼接作为分类分支的输入
        features = torch.cat([shared_features, risk_score], dim=-1)  # 修正：dim=-1
        risk_class_prob = self.classification_branch(features)  # 分类概率
        
        return risk_score, risk_class_prob