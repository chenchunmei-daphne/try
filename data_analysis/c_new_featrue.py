import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.nn.functional as F
from paddle.io import DataLoader, TensorDataset

# 设置随机种子确保结果可复现
paddle.seed(42)
np.random.seed(42)

# 检查设备
device = paddle.get_device()
print(f"Using device: {device}")
paddle.set_device(device)

# 读取数据
data = pd.read_csv("try/group_work/insurance_data_preprocessed.csv")

# 定义我们要使用的特征列表
selected_features = [
    'age', 'bmi', 'systolic_bp', 'diastolic_bp', 'ldl', 'hba1c', 'smoker',  # 基础健康指标

    'chronic_count', 'hypertension', 'diabetes', 'asthma', 'copd', 'cardiovascular_disease',  # 慢性疾病
    'kidney_disease', 'liver_disease',  'cancer_history', 'arthritis', 'mental_health',
    
    'visits_last_year', 'medication_count', 'hospitalizations_last_3yrs', 'days_hospitalized_last_3yrs',  # 医疗利用情况
    # 'risk_score',
    
    'proc_surgery_count', 'had_major_procedure', 'proc_imaging_count', 'proc_physio_count', # 手术/治疗历史
    'proc_consult_count', 'proc_lab_count',

    'sex_Male', 'sex_Other', 'education', 'region_East', 'region_North', 'region_South', 'region_West'] # 人口统计学


# 检查哪些特征在数据集中可用
available_features = []
for feature in selected_features:
    if feature in data.columns:
        available_features.append(feature)
    else:
        print(f"Warning: Feature '{feature}' not found in dataset")

print(f"使用 {len(available_features)} 个特征进行分类")
print(f"可用特征: {available_features}")

# 选择特征和目标变量
X = data[available_features]
y = data['is_high_risk']

# 检查类别分布
print(f"\n类别分布:")
print(y.value_counts())
print(f"高风险比例: {y.mean():.3f}")

# 数据标准化 - 只对数值特征进行标准化
# numerical_features_to_scale = [
#     'age', 'bmi', 'systolic_bp', 'diastolic_bp', 'ldl', 'hba1c',
#     'chronic_count', 'visits_last_year', 'medication_count',
#     'hospitalizations_last_3yrs', 'days_hospitalized_last_3yrs',
#     'risk_score', 'proc_surgery_count', 'proc_imaging_count',
#     'proc_physio_count', 'proc_consult_count', 'proc_lab_count',
#     'education'
# ]

# existing_numerical = [f for f in numerical_features_to_scale if f in X.columns]

# print(f"\n需要标准化的数值特征 ({len(existing_numerical)}个):")
# print(existing_numerical)

# 创建标准化器
# scaler_X = StandardScaler()

# 对数值特征进行标准化
X_scaled = X.copy()
# if existing_numerical:
#     X_scaled[existing_numerical] = scaler_X.fit_transform(X[existing_numerical])
#     print("数值特征已标准化")
# else:
#     print("没有需要标准化的数值特征")

# 划分数据集 - 使用分层抽样
X_train, X_test0, y_train, y_test0 = train_test_split(
    X_scaled, y, 
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_test0, y_test0, test_size=0.5,  
    random_state=42,
    stratify=y_test0
)

print(f"\n数据集大小:")
print(f"训练集: {len(X_train)} 样本 ({len(X_train)/len(X)*100:.1f}%)")
print(f"验证集: {len(X_val)} 样本 ({len(X_val)/len(X)*100:.1f}%)")
print(f"测试集: {len(X_test)} 样本 ({len(X_test)/len(X)*100:.1f}%)")    

print(f"\n训练集类别分布: {y_train.value_counts().to_dict()}")
print(f"验证集类别分布: {y_val.value_counts().to_dict()}")
print(f"测试集类别分布: {y_test.value_counts().to_dict()}")

# 转换为Paddle张量
X_train_tensor = paddle.to_tensor(X_train.values.astype('float32'))
X_val_tensor = paddle.to_tensor(X_val.values.astype('float32'))
X_test_tensor = paddle.to_tensor(X_test.values.astype('float32'))

y_train_tensor = paddle.to_tensor(y_train.values.astype('int64'))
y_val_tensor = paddle.to_tensor(y_val.values.astype('int64'))
y_test_tensor = paddle.to_tensor(y_test.values.astype('int64'))

print(f"\n输入特征维度: {X_train_tensor.shape[1]}")
print(f"训练样本数: {X_train_tensor.shape[0]}")

# 创建数据集和数据加载器
train_dataset = TensorDataset([X_train_tensor, y_train_tensor])
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print(f"\n批次大小: {batch_size}, 每轮训练批次: {len(train_loader)}")

# 定义HighRiskClassifier模型
class HighRiskClassifier(nn.Layer):
    def __init__(self, input_size, hidden_layers=[128, 64], dropout_rate=0.3):
        super(HighRiskClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # 构建隐藏层
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1D(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # 输出层 - 二分类输出
        self.output_layer = nn.Sequential(
            nn.Linear(prev_size, 1),
            nn.Sigmoid()
        )
        
        # 权重初始化 - 修复了PaddlePaddle特有的初始化方式
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Linear):
                # Xavier均匀初始化权重
                nn.initializer.XavierUniform()(layer.weight)
                # 偏置初始化为0 - 修复点：使用nn.initializer.Constant
                if layer.bias is not None:
                    nn.initializer.Constant(value=0.0)(layer.bias)
            elif isinstance(layer, nn.BatchNorm1D):
                # BatchNorm的gamma初始化为1，beta初始化为0
                nn.initializer.Constant(value=1.0)(layer.weight)
                nn.initializer.Constant(value=0.0)(layer.bias)
    
    def forward(self, x):
        features = self.hidden_layers(x)
        probability = self.output_layer(features)
        return probability

input_size = X_train_tensor.shape[1]
model = HighRiskClassifier(input_size=input_size)

# 查看模型结构
# print(f"\n模型结构:")
# print(model)
print(f"总参数数量: {sum(p.numpy().size for p in model.parameters())}")
print(f"可训练参数数量: {sum(p.numpy().size for p in model.parameters() if not p.stop_gradient)}")

# # 计算类别权重以处理不平衡
# pos_weight = len(y_train) / (2 * y_train.sum()) if y_train.sum() > 0 else 1.0
# print(f"\n正样本权重: {pos_weight:.4f}")

# 使用BCE损失
criterion = nn.BCELoss()
optimizer = optim.AdamW(parameters=model.parameters(), learning_rate=0.001, weight_decay=1e-4)

# 训练参数
num_epochs = 5  # 先减少epoch数量，查看训练效果
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

print("\n开始训练...")
best_val_loss = float('inf')
best_model_state = None

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    epoch_train_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch_data in train_loader():
        batch_X, batch_y = batch_data
        
        # 前向传播
        pred_prob = model(batch_X)
        batch_y_float = batch_y.astype('float32').unsqueeze(1)
        
        # 计算损失
        loss = criterion(pred_prob, batch_y_float)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        
        epoch_train_loss += loss
        
        # 计算训练准确率
        pred_labels = (pred_prob > 0.5).astype('float32')
        train_correct += (pred_labels == batch_y_float).astype('float32').sum()
        train_total += batch_y.shape[0]
    
    # 验证阶段
    model.eval()
    with paddle.no_grad():
        val_pred_prob = model(X_val_tensor)
        y_val_float = y_val_tensor.astype('float32').unsqueeze(1)
        val_loss = criterion(val_pred_prob, y_val_float)
        
        # 计算验证准确率
        val_pred_labels = (val_pred_prob > 0.5).astype('float32')
        val_accuracy = (val_pred_labels == y_val_float).astype('float32').mean()
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict()
            # 每10个epoch保存一次最佳模型
            if epoch % 10 == 0:
                paddle.save(model.state_dict(), f"best_model_epoch_{epoch}.pdparams")
    
    # 记录损失和准确率
    train_losses.append(epoch_train_loss / len(train_loader))
    val_losses.append(val_loss.item())
    train_accuracies.append(train_correct / train_total)
    val_accuracies.append(val_accuracy.item())
    
    # 动态调整学习率（简单版本）
    if epoch > 10 and val_losses[-1] > val_losses[-2]:
        current_lr = optimizer.get_lr()
        optimizer.set_lr(current_lr * 0.95)
    
    if (epoch + 1) % 5 == 0 or epoch < 3:
        current_lr = optimizer.get_lr()
        print(f'轮次 [{epoch+1}/{num_epochs}]')
        print(f'  训练损失: {train_losses[-1]:.6f}, 验证损失: {val_losses[-1]:.6f}')
        print(f'  训练准确率: {train_accuracies[-1]:.4f}, 验证准确率: {val_accuracies[-1]:.4f}')
        print(f'  学习率: {current_lr:.6f}')
        print('-' * 50)

print("训练完成!")

# 加载最佳模型
if best_model_state is not None:
    model.set_state_dict(best_model_state)
    print("已加载最佳模型权重")

# 保存最终模型
paddle.save(model.state_dict(), "final_high_risk_classifier.pdparams")
print("最终模型已保存到: final_high_risk_classifier.pdparams")

# 最终评估（使用测试集）
model.eval()
with paddle.no_grad():
    test_pred_prob = model(X_test_tensor)
    
    test_pred_prob_np = test_pred_prob.numpy().flatten()
    test_pred_labels_np = (test_pred_prob_np > 0.5).astype(int)
    y_test_np = y_test_tensor.numpy()

# 计算最终指标
final_accuracy = accuracy_score(y_test_np, test_pred_labels_np)
final_auc = roc_auc_score(y_test_np, test_pred_prob_np)
final_precision = precision_score(y_test_np, test_pred_labels_np, zero_division=0)
final_recall = recall_score(y_test_np, test_pred_labels_np, zero_division=0)
final_f1 = f1_score(y_test_np, test_pred_labels_np, zero_division=0)

print("\n最终测试结果:")
print(f"准确率: {final_accuracy:.6f}")
print(f"AUC: {final_auc:.6f}")
print(f"精确率: {final_precision:.6f}")
print(f"召回率: {final_recall:.6f}")
print(f"F1分数: {final_f1:.6f}")

# 绘制损失曲线和准确率曲线
plt.figure(figsize=(15, 10))

# 损失曲线
plt.subplot(2, 3, 1)
plt.plot(train_losses, label='训练损失', color='blue', alpha=0.7)
plt.plot(val_losses, label='验证损失', color='red', alpha=0.7)
plt.xlabel('训练轮次')
plt.ylabel('BCE损失')
plt.title('训练和验证损失曲线')
plt.legend()
plt.grid(True, alpha=0.3)

# 准确率曲线
plt.subplot(2, 3, 2)
plt.plot(train_accuracies, label='训练准确率', color='blue', alpha=0.7)
plt.plot(val_accuracies, label='验证准确率', color='red', alpha=0.7)
plt.xlabel('训练轮次')
plt.ylabel('准确率')
plt.title('训练和验证准确率曲线')
plt.legend()
plt.grid(True, alpha=0.3)

# ROC曲线
plt.subplot(2, 3, 3)
fpr, tpr, _ = roc_curve(y_test_np, test_pred_prob_np)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {final_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5, label='随机分类器')
plt.xlabel('假正例率')
plt.ylabel('真正例率')
plt.title('ROC曲线')
plt.legend()
plt.grid(True, alpha=0.3)

# 精确率-召回率曲线
plt.subplot(2, 3, 4)
precision, recall, _ = precision_recall_curve(y_test_np, test_pred_prob_np)
plt.plot(recall, precision, color='green', lw=2)
plt.xlabel('召回率')
plt.ylabel('精确率')
plt.title('精确率-召回率曲线')
plt.grid(True, alpha=0.3)

# 混淆矩阵热图
plt.subplot(2, 3, 5)
cm = confusion_matrix(y_test_np, test_pred_labels_np)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['预测低风险', '预测高风险'],
            yticklabels=['真实低风险', '真实高风险'])
plt.title('混淆矩阵')
plt.ylabel('真实标签')
plt.xlabel('预测标签')

plt.tight_layout()
plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 输出详细统计信息
print("\n详细预测统计:")
print(f"预测高风险比例: {test_pred_labels_np.mean():.3f}")
print(f"真实高风险比例: {y_test_np.mean():.3f}")

# 混淆矩阵详细分析
tn, fp, fn, tp = cm.ravel()
print(f"\n混淆矩阵分析:")
print(f"真正例 (TP): {tp}")
print(f"真反例 (TN): {tn}")
print(f"假正例 (FP): {fp}")
print(f"假反例 (FN): {fn}")

print(f"\n性能评估:")
print(f"  准确率: {final_accuracy:.4f}")
print(f"  AUC: {final_auc:.4f}")
print(f"  F1分数: {final_f1:.4f}")

# 保存预测结果
predictions_df = pd.DataFrame({
    'true_label': y_test_np,
    'predicted_prob': test_pred_prob_np,
    'predicted_label': test_pred_labels_np
})

predictions_df.to_csv('high_risk_predictions.csv', index=False)
print("\n预测结果已保存到: high_risk_predictions.csv")

# 特征重要性分析（基于模型权重）
print(f"\n特征重要性分析:")
# 获取第一层的权重作为特征重要性指标
first_layer_weights = model.hidden_layers[0].weight.numpy()
feature_importance = np.abs(first_layer_weights).mean(axis=0)

importance_df = pd.DataFrame({
    'feature': available_features,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"\nTop 10 最重要特征:")
print(importance_df.head(10))

print(f"\nBottom 10 最不重要特征:")
print(importance_df.tail(10))

# 绘制特征重要性图
plt.figure(figsize=(12, 8))
top_features = importance_df.head(20)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('特征重要性（第一层权重绝对值平均）')
plt.title('Top 20 特征重要性')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()