# =========================================================
#  注意力机制文本分类（最终清晰版，Paddle 3.0 + Windows + CPU）
# =========================================================

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset, DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
import os

# =========================================================
# 1. 生成模拟 IMDB 数据（无需下载，训练超快）
# =========================================================

def build_fake_imdb(num_samples=1000, vocab_size=5000, max_len=80):
    """
    随机生成 IMDB 风格文本数据
    """
    data = []
    for _ in range(num_samples):
        seq_len = random.randint(10, max_len)
        text = np.random.randint(2, vocab_size, size=seq_len).tolist()
        label = random.randint(0, 1)
        data.append((text, label))
    return data

train_data = build_fake_imdb(1000)
dev_data   = build_fake_imdb(300)

print("已生成模拟 IMDB 数据（共 1000 条训练样本）")

# =========================================================
# 2. 数据集定义
# =========================================================

class IMDBDataset(Dataset):
    def __init__(self, dataset, pad_id=1, max_len=80):
        self.dataset = dataset
        self.pad_id = pad_id
        self.max_len = max_len

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = x[:self.max_len]
        x = x + [self.pad_id] * (self.max_len - len(x))
        return np.array(x, dtype="int64"), np.array([y], dtype="int64")

    def __len__(self):
        return len(self.dataset)

def collate_fn(batch):
    xs = [b[0] for b in batch]
    ys = [b[1] for b in batch]
    return paddle.to_tensor(xs), paddle.to_tensor(ys)

train_set = IMDBDataset(train_data)
dev_set   = IMDBDataset(dev_data)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
dev_loader   = DataLoader(dev_set,   batch_size=32, shuffle=False, collate_fn=collate_fn)

print("数据加载器已构建完成")

# =========================================================
# 3. 注意力机制（点积 Attention）
# =========================================================

class Attention(nn.Layer):
    def __init__(self, hidden_size):
        super().__init__()
        # 查询向量 q：可学习参数
        self.q = paddle.create_parameter(
            shape=[hidden_size, 1],
            dtype="float32",
            default_initializer=nn.initializer.Uniform(-0.1, 0.1)
        )

    def forward(self, x):
        # x: [B, L, H]
        scores = paddle.matmul(x, self.q)  # [B,L,1]
        scores = scores.squeeze(-1)        # [B,L]

        weights = F.softmax(scores, axis=1)  # [B,L]

        # context = Σ α_i * h_i
        context = paddle.matmul(weights.unsqueeze(1), x)  # [B,1,H]
        return context.squeeze(1), weights

# =========================================================
# 4. BiLSTM + Attention 文本分类模型
# =========================================================

class BiLSTMAtt(nn.Layer):
    def __init__(self, vocab_size=5000, emb_size=64, hidden_size=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, direction="bidirectional")
        self.att = Attention(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, x):
        emb = self.embed(x)
        h, _ = self.lstm(emb)
        att_out, weights = self.att(h)
        logits = self.fc(att_out)
        return logits, weights

# =========================================================
# 5. 训练函数（增加损失和准确率记录）
# =========================================================

def train(model, loader, dev_loader, epochs=3):
    opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # 用于记录训练过程中的指标
    train_losses = []  # 每个epoch的平均训练损失
    dev_accuracies = []  # 每个epoch的验证集准确率
    step_losses = []  # 每个step的损失（用于详细曲线）
    step_indices = []  # step对应的索引
    
    step_count = 0
    
    for epoch in range(epochs):
        epoch_losses = []  # 当前epoch的所有step损失
        
        for step, (x, y) in enumerate(loader):
            logits, _ = model(x)
            loss = criterion(logits, y.squeeze(1))
            loss.backward()
            opt.step()
            opt.clear_grad()
            
            # 记录step损失
            step_loss = float(loss.numpy())
            epoch_losses.append(step_loss)
            step_losses.append(step_loss)
            step_indices.append(step_count)
            step_count += 1
            
            if step % 20 == 0:
                print(f"epoch {epoch} step {step} loss={step_loss:.4f}")
        
        # 计算当前epoch的平均训练损失
        avg_epoch_loss = np.mean(epoch_losses)
        train_losses.append(avg_epoch_loss)
        
        # 评估验证集准确率
        acc = evaluate(model, dev_loader)
        dev_accuracies.append(acc)
        
        print(f"epoch={epoch} avg_loss={avg_epoch_loss:.4f} dev_acc={acc:.4f}\n")
    
    # 返回记录的数据用于可视化
    return {
        'train_losses': train_losses,
        'dev_accuracies': dev_accuracies,
        'step_losses': step_losses,
        'step_indices': step_indices,
        'epochs': epochs
    }

def evaluate(model, loader):
    correct, total = 0, 0
    model.eval()
    with paddle.no_grad():
        for x, y in loader:
            logits, _ = model(x)
            pred = logits.argmax(axis=1)
            correct += (pred == y.squeeze(1)).numpy().sum()
            total += len(y)
    model.train()
    return correct / total

# =========================================================
# 6. 可视化函数
# =========================================================

def plot_training_history(history):
    """
    绘制训练过程中的损失和准确率曲线
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 每个epoch的平均训练损失
    axes[0, 0].plot(range(1, len(history['train_losses']) + 1), history['train_losses'], 
                   marker='o', linewidth=2, markersize=8, color='red')
    axes[0, 0].set_title('Training Loss per Epoch', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(range(1, len(history['train_losses']) + 1))
    
    # 2. 验证集准确率
    axes[0, 1].plot(range(1, len(history['dev_accuracies']) + 1), history['dev_accuracies'], 
                   marker='s', linewidth=2, markersize=8, color='green')
    axes[0, 1].set_title('Validation Accuracy per Epoch', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].set_ylim([0, 1.05])
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(range(1, len(history['dev_accuracies']) + 1))
    
    # 3. 每个step的详细损失曲线
    axes[1, 0].plot(history['step_indices'], history['step_losses'], 
                   linewidth=1, alpha=0.7, color='blue')
    axes[1, 0].set_title('Training Loss per Step', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Step', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 损失和准确率对比图（双y轴）
    ax1 = axes[1, 1]
    epochs = range(1, len(history['train_losses']) + 1)
    
    # 绘制损失曲线（左轴）
    color = 'tab:red'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', color=color, fontsize=12)
    line1 = ax1.plot(epochs, history['train_losses'], marker='o', 
                    color=color, linewidth=2, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # 绘制准确率曲线（右轴）
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Accuracy', color=color, fontsize=12)
    line2 = ax2.plot(epochs, history['dev_accuracies'], marker='s', 
                    color=color, linewidth=2, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, 1.05])
    
    # 添加图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center')
    
    axes[1, 1].set_title('Loss & Accuracy Comparison', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("训练历史图已保存：training_history.png")

def visualize_attention(model):
    """
    可视化注意力权重
    """
    sample = train_data[0][0][:80]
    sample = sample + [1] * (80 - len(sample))
    x = paddle.to_tensor([sample], dtype="int64")

    logits, att = model(x)
    att = att.numpy()[0]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    
    # 1. 条形图
    axes[0].bar(range(len(att)), att, color='skyblue', edgecolor='navy', alpha=0.7)
    axes[0].set_title("Attention Weights (Bar Chart)", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Token Position", fontsize=12)
    axes[0].set_ylabel("Attention Weight", fontsize=12)
    axes[0].grid(True, axis='y', alpha=0.3)
    
    # 2. 折线图
    axes[1].plot(range(len(att)), att, marker='o', color='red', 
                linewidth=2, markersize=4)
    axes[1].fill_between(range(len(att)), att, alpha=0.3, color='red')
    axes[1].set_title("Attention Weights (Line Chart)", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Token Position", fontsize=12)
    axes[1].set_ylabel("Attention Weight", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("attention_vis.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("注意力可视化已保存：attention_vis.png")

# =========================================================
# 7. 主流程
# =========================================================

def main():
    print("=" * 60)
    print("注意力机制文本分类实验")
    print("=" * 60)
    
    # 创建保存图片的目录（如果不存在）
    os.makedirs('.', exist_ok=True)
    
    print("\n1. 构建模型 ...")
    model = BiLSTMAtt()
    
    print("\n2. 开始训练 ...")
    print("-" * 40)
    history = train(model, train_loader, dev_loader, epochs=30)
    
    print("\n3. 绘制训练历史 ...")
    print("-" * 40)
    plot_training_history(history)
    
    print("\n4. 可视化注意力 ...")
    print("-" * 40)
    visualize_attention(model)
    
    print("\n" + "=" * 60)
    print("实验全部完成！")
    print("已生成以下可视化文件：")
    print("1. training_history.png - 训练损失和准确率曲线")
    print("2. attention_vis.png - 注意力权重可视化")
    print("=" * 60)

if __name__ == "__main__":
    main()