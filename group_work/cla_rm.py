import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def advanced_preprocessing(df, target_col='plan_type'):
    """
    高级数据预处理函数 - 区分有序变量和名义变量
    """
    data = df.copy()
    
    print("开始高级数据预处理...")
    print(f"原始数据形状: {data.shape}")
    
    # 1. 删除无关特征
    if 'person_id' in data.columns:
        data = data.drop('person_id', axis=1)
        print("已删除 person_id 列")
    
    # 2. 处理缺失值
    missing_cols = data.columns[data.isnull().any()].tolist()
    print(f"\n有缺失值的列: {missing_cols}")
    
    for col in missing_cols:
        if data[col].dtype == 'object':
            mode_val = data[col].mode()[0]
            data[col].fillna(mode_val, inplace=True)
            print(f"分类特征 '{col}' 用众数 '{mode_val}' 填充")
        else:
            median_val = data[col].median()
            data[col].fillna(median_val, inplace=True)
            print(f"数值特征 '{col}' 用中位数 {median_val} 填充")
    
    # 3. 定义编码策略
    ordinal_features = {
        'education': ['No HS', 'HS', 'Some College', 'Bachelors', 'Masters', 'Doctorate'],
        'smoker': ['Never', 'Former', 'Current'],
        'alcohol_freq': ['Occasional', 'Weekly', 'Daily'], 
        'network_tier': ['Bronze', 'Silver', 'Gold', 'Platinum']
    }
    
    # 名义变量（无序分类变量）
    nominal_features = ['sex', 'region', 'urban_rural', 'marital_status', 'employment_status']
    
    # 4. 分离特征和目标变量
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    
    # 5. 识别数值特征
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # 从数值特征中移除已经在有序/名义变量中定义的特征
    numerical_cols = [col for col in numerical_cols if col not in ordinal_features and col not in nominal_features]
    
    print(f"\n特征类型分布:")
    print(f"  有序变量: {len(ordinal_features)}个 - {list(ordinal_features.keys())}")
    print(f"  名义变量: {len(nominal_features)}个 - {nominal_features}")
    print(f"  数值特征: {len(numerical_cols)}个")
    print(f"  目标变量: 1个 - {target_col}")
    
    # 6. 有序变量编码
    print(f"\n=== 有序变量编码 ===")
    ordinal_encoder = OrdinalEncoder(
        categories=[ordinal_features[feat] for feat in ordinal_features.keys()], 
        dtype=np.int64
    )
    
    # 检查并处理有序变量中可能缺失的类别
    for feat in ordinal_features.keys():
        unique_vals = set(X[feat].unique())
        expected_vals = set(ordinal_features[feat])
        missing_in_data = expected_vals - unique_vals
        if missing_in_data:
            print(f"  警告: {feat} 缺少类别: {missing_in_data}")
    
    X[list(ordinal_features.keys())] = ordinal_encoder.fit_transform(X[list(ordinal_features.keys())])
    
    for feat in ordinal_features.keys():
        print(f"  {feat}: {ordinal_features[feat]} -> 编码为 0-{len(ordinal_features[feat])-1}")
    
    # 7. 名义变量独热编码
    print(f"\n=== 名义变量独热编码 ===")
    # 先创建副本，避免SettingWithCopyWarning
    X_encoded = X.copy()
    
    # 对每个名义变量进行独热编码
    for col in nominal_features:
        dummies = pd.get_dummies(X[col], prefix=col, drop_first=True, dtype=np.int64)
        X_encoded = pd.concat([X_encoded, dummies], axis=1)
        X_encoded = X_encoded.drop(col, axis=1)
        print(f"  {col}: 创建了 {dummies.shape[1]} 个虚拟变量")
    
    # # 8. 标准化数值特征
    # print(f"\n=== 数值特征标准化 ===")
    # scaler = StandardScaler()
    # X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])
    # print(f"  标准化了 {len(numerical_cols)} 个数值特征")
    
    # 9. 编码目标变量
    print(f"\n=== 目标变量编码 ===")
    # 目标变量作为名义变量处理，使用LabelEncoder
    from sklearn.preprocessing import LabelEncoder
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    print(f"  目标变量 '{target_col}' 编码完成: {len(le_target.classes_)} 个类别")
    print(f"  类别分布: {dict(zip(le_target.classes_, np.bincount(y_encoded)))}")
    
    # 10. 最终数据检查
    print(f"\n=== 最终数据检查 ===")
    print(f"  最终特征数量: {X_encoded.shape[1]}")
    print(f"  样本数量: {X_encoded.shape[0]}")
    
    # 检查是否有object类型剩余
    remaining_object_cols = X_encoded.select_dtypes(include=['object']).columns.tolist()
    if remaining_object_cols:
        print(f"  警告: 仍有object类型特征: {remaining_object_cols}")
    else:
        print(f" 所有特征都已正确编码，无object类型剩余")
    
    # 返回所有预处理组件
    preprocessing_info = {
        'ordinal_features': ordinal_features,
        'nominal_features': nominal_features,
        'numerical_cols': numerical_cols,
        'ordinal_encoder': ordinal_encoder,
        # 'scaler': scaler,
        'le_target': le_target
    }
    
    return X_encoded, y_encoded, preprocessing_info

# 执行高级预处理
df = pd.read_csv('medical_insurance.csv')
df = df[df['age'] > 20]  # 删除异常数据
X_advanced, y_advanced, preprocess_info = advanced_preprocessing(df)

test_size = 0.8
X_train, X_test, y_train, y_test = train_test_split( X_advanced, y_advanced, test_size=test_size,
                                                    random_state=42, stratify=y_advanced)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 使用训练集的参数来转换测试集


y_advanced = pd.Series(y_advanced, name='plan_type')

# Calculate correlation between all features and target variable
# correlation_with_target = X_advanced.corrwith(y_advanced).sort_values(ascending=False)

# print("Feature correlations with annual_medical_cost (sorted):")
# print(correlation_with_target)

# # Plot correlation bar chart
# plt.figure(figsize=(12, 20))
# correlation_with_target.plot(kind='barh')
# plt.title('Feature Correlations with annual_medical_cost')
# plt.xlabel('Correlation Coefficient')
# plt.tight_layout()
# plt.show()