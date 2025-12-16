import paddle
import paddle.nn as nn
import paddle.optimizer as optimizer
import paddle.io as io
import paddle.nn.functional as F
from paddle.vision.datasets import MNIST
from paddle.vision.models import LeNet
import matplotlib.pyplot as plt
import numpy as np

# 自动选择设备（优先使用GPU）
if paddle.is_compiled_with_cuda():
    paddle.set_device('gpu:1')
    print("✓ 使用 GPU 训练")
else:
    paddle.set_device('cpu')
    print("✓ 使用 CPU 训练")
paddle.seed(0)

# ==================== 数据预处理 ====================
def transform(image):
    """图像规范化到0~1之间"""
    image = paddle.to_tensor(image / 255, dtype='float32')
    image = paddle.unsqueeze(image, axis=0)
    return image

# ==================== 改造后的LeNet模型 ====================
class ImprovedLeNet(nn.Layer):
    """
    改造后的LeNet模型，加入了Dropout和BatchNorm
    """
    def __init__(self, num_classes=10, dropout_rate=0.5, use_batchnorm=True):
        super(ImprovedLeNet, self).__init__()
        
        self.use_batchnorm = use_batchnorm
        
        # 卷积层1: 1 -> 6
        self.conv1 = nn.Conv2D(1, 6, 5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2D(6) if use_batchnorm else None
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2)
        
        # 卷积层2: 6 -> 16
        self.conv2 = nn.Conv2D(6, 16, 5, stride=1)
        self.bn2 = nn.BatchNorm2D(16) if use_batchnorm else None
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(400, 120)
        self.bn3 = nn.BatchNorm1D(120) if use_batchnorm else None
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1D(84) if use_batchnorm else None
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        # 第一层卷积 + BN + 激活 + 池化
        x = self.conv1(x)
        if self.use_batchnorm and self.bn1:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # 第二层卷积 + BN + 激活 + 池化
        x = self.conv2(x)
        if self.use_batchnorm and self.bn2:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # 展平
        x = paddle.flatten(x, 1)
        
        # 全连接层1 + BN + 激活 + Dropout
        x = self.fc1(x)
        if self.use_batchnorm and self.bn3:
            x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # 全连接层2 + BN + 激活 + Dropout
        x = self.fc2(x)
        if self.use_batchnorm and self.bn4:
            x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # 输出层
        x = self.fc3(x)
        
        return x


# ==================== 使用LayerNorm的版本 ====================
class ImprovedLeNetLayerNorm(nn.Layer):
    """
    改造后的LeNet模型，使用LayerNorm而不是BatchNorm
    """
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(ImprovedLeNetLayerNorm, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2D(1, 6, 5, stride=1, padding=2)
        self.ln1 = nn.LayerNorm([6, 28, 28])
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2D(6, 16, 5, stride=1)
        self.ln2 = nn.LayerNorm([16, 10, 10])
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(400, 120)
        self.ln3 = nn.LayerNorm(120)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(120, 84)
        self.ln4 = nn.LayerNorm(84)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = paddle.flatten(x, 1)
        
        x = self.fc1(x)
        x = self.ln3(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.ln4(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


# ==================== 训练函数 ====================
def train_model(model, train_loader, opt, loss_fn, num_epochs=10):
    """训练模型并记录损失"""
    model.train()
    epoch_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        
        for batch_id, data in enumerate(train_loader):
            x, y = data
            
            # 前向传播
            logits = model(x)
            loss = loss_fn(logits, y)
            
            # 反向传播
            loss.backward()
            
            # 参数更新
            opt.step()
            opt.clear_grad()
            
            total_loss += float(loss.numpy())
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        epoch_losses.append(avg_loss)
        
        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return epoch_losses


# ==================== 评估函数 ====================
@paddle.no_grad()
def evaluate_model(model, test_loader):
    """评估模型准确率"""
    model.eval()
    correct = 0
    total = 0
    
    for data in test_loader:
        x, y = data
        # 确保标签形状正确
        if len(y.shape) > 1:
            y = paddle.squeeze(y, axis=-1)
        
        logits = model(x)
        pred = paddle.argmax(logits, axis=1)
        
        # 确保 pred 和 y 形状一致
        correct += float((pred == y).astype('float32').sum().numpy())
        total += y.shape[0]
    
    accuracy = correct / total
    return accuracy


# ==================== 主实验函数 ====================
def run_experiments():
    """运行所有实验配置"""
    
    print("=" * 60)
    print("开始MNIST模型改造实验")
    print("=" * 60)
    
    # 准备数据
    paddle.vision.image.set_image_backend('cv2')
    train_dataset = MNIST(mode='train', transform=transform)
    test_dataset = MNIST(mode='test', transform=transform)
    
    # 实验配置
    configs = [
        # 配置1: 基础LeNet (对照组)
        {
            'name': 'Baseline LeNet (SGD)',
            'model': LeNet(num_classes=10),
            'optimizer': 'SGD',
            'lr': 0.01,
            'batch_size': 64,
            'color': '#9c9d9f'
        },
        # 配置2: 改进版LeNet + BatchNorm + Dropout + SGD
        {
            'name': 'LeNet + BN + Dropout (SGD)',
            'model': ImprovedLeNet(dropout_rate=0.3, use_batchnorm=True),
            'optimizer': 'SGD',
            'lr': 0.01,
            'batch_size': 64,
            'color': '#f7d2e2'
        },
        # 配置3: 改进版LeNet + BatchNorm + Dropout + Adam
        {
            'name': 'LeNet + BN + Dropout (Adam)',
            'model': ImprovedLeNet(dropout_rate=0.3, use_batchnorm=True),
            'optimizer': 'Adam',
            'lr': 0.001,
            'batch_size': 64,
            'color': '#f19ec2'
        },
        # 配置4: 改进版LeNet + BatchNorm + Dropout + RMSprop
        {
            'name': 'LeNet + BN + Dropout (RMSprop)',
            'model': ImprovedLeNet(dropout_rate=0.3, use_batchnorm=True),
            'optimizer': 'RMSprop',
            'lr': 0.001,
            'batch_size': 64,
            'color': '#e86096'
        },
        # 配置5: 改进版LeNet + LayerNorm + Dropout + Adam
        {
            'name': 'LeNet + LayerNorm + Dropout (Adam)',
            'model': ImprovedLeNetLayerNorm(dropout_rate=0.3),
            'optimizer': 'Adam',
            'lr': 0.001,
            'batch_size': 64,
            'color': '#000000'
        },
    ]
    
    results = []
    num_epochs = 20
    
    # 运行每个配置
    for idx, config in enumerate(configs):
        print(f"\n{'-'*60}")
        print(f"实验 {idx+1}/{len(configs)}: {config['name']}")
        print(f"{'-'*60}")
        
        # 重置随机种子保证可复现性
        paddle.seed(0)
        
        # 准备数据加载器
        train_loader = io.DataLoader(
            train_dataset, 
            batch_size=config['batch_size'],
            shuffle=False  # 为保证结果一致性，不打乱
        )
        test_loader = io.DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False
        )
        
        # 创建优化器
        if config['optimizer'] == 'SGD':
            opt = optimizer.SGD(
                learning_rate=config['lr'],
                parameters=config['model'].parameters()
            )
        elif config['optimizer'] == 'Adam':
            opt = optimizer.Adam(
                learning_rate=config['lr'],
                parameters=config['model'].parameters()
            )
        elif config['optimizer'] == 'RMSprop':
            opt = optimizer.RMSProp(
                learning_rate=config['lr'],
                parameters=config['model'].parameters()
            )
        elif config['optimizer'] == 'AdaGrad':
            opt = optimizer.Adagrad(
                learning_rate=config['lr'],
                parameters=config['model'].parameters()
            )
        
        # 训练模型
        loss_fn = F.cross_entropy
        epoch_losses = train_model(
            config['model'],
            train_loader,
            opt,
            loss_fn,
            num_epochs=num_epochs
        )
        
        # 评估模型
        accuracy = evaluate_model(config['model'], test_loader)
        
        print(f"最终测试准确率: {accuracy*100:.2f}%")
        
        # 保存结果
        results.append({
            'name': config['name'],
            'losses': epoch_losses,
            'accuracy': accuracy,
            'color': config['color']
        })
    
    return results


# ==================== 可视化函数 ====================
def plot_results(results):
    """绘制训练结果对比图"""
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    for result in results:
        ax1.plot(
            result['losses'],
            label=result['name'],
            color=result['color'],
            linewidth=2
        )
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 绘制准确率对比
    names = [r['name'] for r in results]
    accuracies = [r['accuracy'] * 100 for r in results]
    colors = [r['color'] for r in results]
    
    bars = ax2.bar(range(len(names)), accuracies, color=colors, alpha=0.7)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax2.set_ylim([85, 100])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上标注数值
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = '/hsiam02/huotao/Github/practice-in-paddle/chap7网络优化与正则化/mnist_improved_results.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n结果图片已保存至: {save_path}")
    
    plt.show()


# ==================== 生成实验报告 ====================
def generate_report(results):
    """生成实验结果文本报告"""
    report_path = '/hsiam02/huotao/Github/practice-in-paddle/chap7网络优化与正则化/实验结果报告.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MNIST模型改造实验报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("一、实验目的\n")
        f.write("-" * 80 + "\n")
        f.write("改造MNIST模型，加入Dropout层和逐层规范化，对比不同优化算法的效果。\n\n")
        
        f.write("二、实验配置\n")
        f.write("-" * 80 + "\n")
        for idx, result in enumerate(results, 1):
            f.write(f"{idx}. {result['name']}\n")
        f.write("\n")
        
        f.write("三、实验结果\n")
        f.write("-" * 80 + "\n")
        for idx, result in enumerate(results, 1):
            f.write(f"{idx}. {result['name']}\n")
            f.write(f"   - 最终损失: {result['losses'][-1]:.4f}\n")
            f.write(f"   - 测试准确率: {result['accuracy']*100:.2f}%\n")
            f.write(f"   - 初始损失: {result['losses'][0]:.4f}\n")
            f.write(f"   - 损失下降: {result['losses'][0] - result['losses'][-1]:.4f}\n\n")
        
        f.write("四、结论分析\n")
        f.write("-" * 80 + "\n")
        
        # 找出最佳配置
        best_result = max(results, key=lambda x: x['accuracy'])
        f.write(f"1. 最佳配置: {best_result['name']}\n")
        f.write(f"   准确率: {best_result['accuracy']*100:.2f}%\n\n")
        
        f.write("2. 改进效果:\n")
        baseline_acc = results[0]['accuracy']
        f.write(f"   - 基础模型准确率: {baseline_acc*100:.2f}%\n")
        f.write(f"   - 最佳改进模型准确率: {best_result['accuracy']*100:.2f}%\n")
        f.write(f"   - 提升: {(best_result['accuracy']-baseline_acc)*100:.2f}%\n\n")
        
        f.write("3. 方法总结:\n")
        f.write("   - Dropout有效防止过拟合，提高泛化能力\n")
        f.write("   - BatchNorm/LayerNorm加速收敛，稳定训练\n")
        f.write("   - Adam等自适应优化器通常优于传统SGD\n")
        f.write("   - 不同优化器适用于不同场景，需要实验对比\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"实验报告已保存至: {report_path}")


# ==================== 主函数 ====================
if __name__ == '__main__':
    print("开始MNIST模型改造实验...")
    print("这可能需要几分钟时间，请耐心等待...\n")
    
    # 运行实验
    results = run_experiments()
    
    # 可视化结果
    print("\n正在生成结果图表...")
    plot_results(results)
    
    # 生成报告
    print("\n正在生成实验报告...")
    generate_report(results)
    
    print("\n" + "="*60)
    print("实验完成！")
    print("="*60)
    print("\n生成的文件:")
    print("1. mnist_improved_results.png - 训练结果对比图")
    print("2. 实验结果报告.txt - 详细实验报告")



'''
实验结果，py 文件无法显示结果图

================================================================================
MNIST模型改造实验报告
================================================================================

一、实验目的
--------------------------------------------------------------------------------
改造MNIST模型，加入Dropout层和逐层规范化，对比不同优化算法的效果。

二、实验配置
--------------------------------------------------------------------------------
1. Baseline LeNet (SGD)
2. LeNet + BN + Dropout (SGD)
3. LeNet + BN + Dropout (Adam)
4. LeNet + BN + Dropout (RMSprop)
5. LeNet + LayerNorm + Dropout (Adam)

三、实验结果
--------------------------------------------------------------------------------
1. Baseline LeNet (SGD)
   - 最终损失: 0.0387
   - 测试准确率: 98.56%
   - 初始损失: 0.3721
   - 损失下降: 0.3334

2. LeNet + BN + Dropout (SGD)
   - 最终损失: 0.0716
   - 测试准确率: 98.76%
   - 初始损失: 0.7573
   - 损失下降: 0.6856

3. LeNet + BN + Dropout (Adam)
   - 最终损失: 0.0157
   - 测试准确率: 99.10%
   - 初始损失: 0.2952
   - 损失下降: 0.2795

4. LeNet + BN + Dropout (RMSprop)
   - 最终损失: 0.0169
   - 测试准确率: 99.00%
   - 初始损失: 0.2464
   - 损失下降: 0.2296

5. LeNet + LayerNorm + Dropout (Adam)
   - 最终损失: 0.0176
   - 测试准确率: 98.99%
   - 初始损失: 0.3311
   - 损失下降: 0.3135

四、结论分析
--------------------------------------------------------------------------------
1. 最佳配置: LeNet + BN + Dropout (Adam)
   准确率: 99.10%

2. 改进效果:
   - 基础模型准确率: 98.56%
   - 最佳改进模型准确率: 99.10%
   - 提升: 0.54%

3. 方法总结:
   - Dropout有效防止过拟合，提高泛化能力
   - BatchNorm/LayerNorm加速收敛，稳定训练
   - Adam等自适应优化器通常优于传统SGD
   - 不同优化器适用于不同场景，需要实验对比

================================================================================




'''