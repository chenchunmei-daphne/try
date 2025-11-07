import torch
# 准备分类模型
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

classification_models = {
    "Decision Tree Classification": DecisionTreeClassifier(random_state=42),
    "MLP Classification": MLPClassifier(hidden_layer_sizes=(50, 25), activation='relu', solver='adam', 
                                      batch_size=32, max_iter=1000, random_state=42),
    "K-Neighbors Classification": KNeighborsClassifier(n_neighbors=5)
}

flag_standard_classification = True  # 是否标准化特征

def classification_train(X_train, X_test, models, y_train=y_train, y_test=y_test):
    n_models = len(models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (name, model) in enumerate(models.items()):
        time_start = time.time()
        model.fit(X_train, y_train)
        
        # 训练集预测和评估
        y_train_pred = model.predict(X_train)
        accuracy_train = accuracy_score(y_train, y_train_pred)
        precision_train = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
        recall_train = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
        f1_train = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
        
        # 测试集预测和评估
        predictions = model.predict(X_test)
        accuracy_test = accuracy_score(y_test, predictions)
        precision_test = precision_score(y_test, predictions, average='weighted', zero_division=0)
        recall_test = recall_score(y_test, predictions, average='weighted', zero_division=0)
        f1_test = f1_score(y_test, predictions, average='weighted', zero_division=0)
        
        # 绘制混淆矩阵
        cm = confusion_matrix(y_test, predictions)
        ax = axes[idx]
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'Confusion Matrix - {name}')
        plt.colorbar(im, ax=ax)
        
        # 添加数值标签
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
        print("Model:", name)
        print("模型在训练集上的评估结果：")
        print("Accuracy:", accuracy_train)
        print("Precision:", precision_train)
        print("Recall:", recall_train)
        print("F1 Score:", f1_train)
        print()
        print("模型在测试集上的评估结果：")
        print("Accuracy:", accuracy_test)
        print("Precision:", precision_test)
        print("Recall:", recall_test)
        print("F1 Score:", f1_test)
        print("\n分类报告:")
        print(classification_report(y_test, predictions, zero_division=0))
        print('-'*50)
        time_end = time.time()
        print(f"训练和预测时间: {time_end - time_start:.2f}秒")
        print()

# 使用全部的特征进行训练并评估分类模型
print("Classification Models Evaluation with all features:")
if flag_standard_classification:
    scaler2 = StandardScaler()
    X_train2 = scaler2.fit_transform(X_train)
    X_test2 = scaler2.transform(X_test)
    
classification_train(X_train=X_train2, X_test=X_test2, models=classification_models)