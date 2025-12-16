import paddle
import paddle.nn as nn
import numpy as np
import re
import jieba
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 设置随机种子
paddle.seed(102)
np.random.seed(102)


# --------------------------
# 1. 数据读取与预处理（优化）
# --------------------------
class TextProcessor:
    def __init__(self, max_vocab_size=5000, max_seq_len=100):
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len
        self.word2id = {'<PAD>': 0, '<UNK>': 1}
        self.id2word = {0: '<PAD>', 1: '<UNK>'}
        self.label2id = {0: 0, 1: 1}  # 固定标签映射，避免样本少导致映射混乱
        self.id2label = {0: 0, 1: 1}

    def _clean_text(self, text):
        """保留情感相关符号（如！、？）"""
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9！？，。]', ' ', text)
        return text.strip()

    def _tokenize(self, text):
        return jieba.lcut(text)

    def build_vocab(self, texts):
        word_counts = Counter()
        for text in texts:
            cleaned = self._clean_text(text)
            tokens = self._tokenize(cleaned)
            word_counts.update(tokens)
        top_words = word_counts.most_common(self.max_vocab_size - 2)
        for i, (word, _) in enumerate(top_words, 2):
            self.word2id[word] = i
            self.id2word[i] = word

    def text_to_ids(self, text):
        cleaned = self._clean_text(text)
        tokens = self._tokenize(cleaned)
        ids = [self.word2id.get(token, self.word2id['<UNK>']) for token in tokens]
        if len(ids) > self.max_seq_len:
            ids = ids[:self.max_seq_len]
        else:
            ids += [self.word2id['<PAD>']] * (self.max_seq_len - len(ids))
        return ids

    def process_dataset(self, texts, labels=None):
        text_ids = [self.text_to_ids(text) for text in texts]
        text_ids = np.array(text_ids, dtype=np.int64)
        if labels is not None:
            label_ids = np.array(labels, dtype=np.int64)
            return text_ids, label_ids
        return text_ids


# 扩展奶茶评价数据集（100条，正负各50条）
def load_sample_data():
    # 正面评价（50条）
    positive = [
        "这款奶茶太好喝了，茶底清香，奶味浓郁，甜度刚好，强烈推荐！",
        "口感顺滑，果肉新鲜，性价比很高，会无限回购的一款奶茶",
        "茶底很正宗，小料给得足，冰度恰到好处，整体体验很棒",
        "这是我喝过最好喝的奶茶，芝士奶盖咸香浓郁，茶香回甘",
        "珍珠Q弹有嚼劲，茶底清爽不苦涩，夏天喝超解暑",
        "奶盖绵密，茶香清新，甜度可以调整，很贴心",
        "喝完回味无穷，茶底醇厚，奶味纯正，值得一试",
        "第一次喝就爱上了，味道层次丰富，小料种类多，超满足",
        "用料扎实，喝得出真材实料，没有添加剂的怪味，放心喝",
        "包装精致，送得很快，拿到手还是热的，口感满分",
        # 新增40条正面评价（省略，实际运行时可补充类似句式）
        "甜度刚刚好，不会腻，茶味很清新，喝完很舒服",
        "小料给得特别多，每一口都能吃到，太满足了",
        "价格实惠，味道不输大牌，性价比超高",
        "奶盖甜咸适中，和茶底融合得很好，强烈推荐",
        "冰沙细腻，夏天喝特别解渴，味道很清爽",
        "店员服务很好，会主动询问甜度和冰度，体验很棒",
        "新品奶茶太惊艳了，果香和茶香完美结合，值得尝试",
        "超大一杯，分量很足，两个人喝都够，性价比高",
        "茶底是现泡的，喝得出新鲜度，味道很纯正",
        "珍珠煮得很到位，Q弹有嚼劲，甜度也刚好",
        "每次路过都会买，已经成了我的固定奶茶店，味道稳定",
        "热饮口感更棒，奶味更浓郁，冬天喝特别舒服",
        "半糖甜度刚好，不会太甜，适合喜欢清淡口味的人",
        "水果很新鲜，搭配茶底很清爽，夏天喝太合适了",
        "芝士奶盖厚厚的，咸香浓郁，和茶底绝配",
        "包装很用心，不会洒出来，细节做得很好",
        "价格虽然偏高，但味道和用料都值这个价",
        "茶味很浓，奶味适中，整体口感很清爽，不腻",
        "小料很有特色，是别家没有的，增加了口感层次",
        "喝了很多年，味道一直没变，品质很稳定",
        "推荐少冰款，不会冲淡味道，口感更好",
        "奶泡细腻，甜度适中，入口很顺滑，体验很好",
        "茶底回甘明显，品质很好，喝得出是好茶叶",
        "配料可以自由搭配，选择很多，很人性化",
        "大杯量很足，性价比高，学生党表示很满意",
        "冷藏后更好喝，夏天放冰箱里，解渴又美味",
        "甜度可以精准调整，对于控糖人士很友好",
        "新品上市很快，总能尝试到新口味，很有新意",
        "虽然排队久，但味道值得，每次都愿意等",
        "外卖包装很好，不会漏，拿到手和店里喝一样",
        "茶香很突出，奶味不抢镜，平衡得很好",
        "珍珠每天现煮，很新鲜，没有硬心，口感好",
        "水果茶里的水果给得很多，喝完还能吃水果，划算",
        "热奶茶加珍珠是绝配，冬天喝暖心又暖胃",
        "甜度不高，适合不爱吃甜的人，味道很清爽",
        "奶盖上面撒了坚果碎，增加了口感，很有创意",
        "茶底选择多，有红茶、绿茶、乌龙茶，满足不同口味",
        "小料都是自己做的，很干净卫生，吃得放心",
        "价格亲民，味道好，已经推荐给身边很多朋友了"
    ]

    # 负面评价（50条）
    negative = [
        "太难喝了，甜得发腻，茶味苦涩，珍珠还硬邦邦的，踩雷了！",
        "完全不符合预期，分量少，价格贵，味道寡淡，不建议购买",
        "包装简陋，拿到手都洒了，味道也一般，不会再买了",
        "非常失望，奶味奇怪，像是过期了，喝了一口就扔了",
        "一般般，没什么特别的，甜度偏高，喝完有点腻",
        "太难喝了，味道奇怪，像是兑了水，性价比极低",
        "料很少，价格贵，味道普通，不如其他家的好喝",
        "珍珠硬得像石头，根本咬不动，太影响口感了",
        "甜得齁人，喝了一口就受不了，直接扔了",
        "茶底有股怪味，像是变质了，太失望了",
        # 新增40条负面评价（省略，实际运行时可补充类似句式）
        "奶味很淡，茶味也很寡，像喝白开水一样，不值这个价",
        "冰块太多，半杯都是冰，喝不了几口就没了，坑人",
        "小料都沉底了，而且很硬，根本嚼不动，差评",
        "价格涨了但分量少了，味道也不如以前，不会再买了",
        "外卖送了一个小时，拿到手都凉了，口感差很多",
        "点的少糖还是甜得发腻，怀疑店员没按要求做",
        "茶底有股焦味，像是煮糊了，太难喝了",
        "包装很差，打开的时候洒了一身，体验极差",
        "奶盖稀得像水一样，一点都不浓郁，差评",
        "水果不新鲜，有点烂掉的味道，太影响口感了",
        "排队半小时，买到的奶茶味道很普通，不值得",
        "热奶茶温度不够，温温的，喝着没感觉，差评",
        "甜度无法调整，对于不爱甜的人太不友好了",
        "茶底太苦了，盖过了奶味，根本喝不下去",
        "小料种类很少，选择不多，而且不新鲜",
        "分量越来越少，价格越来越高，性价比太低",
        "奶茶里有异物，像是头发，太恶心了，再也不买",
        "味道和图片差距太大，实物看着就没食欲",
        "冰块化得太快，没几分钟就变淡了，不好喝",
        "店员态度很差，问问题不耐烦，体验不好",
        "新品太难喝了，完全是噱头，浪费钱",
        "奶味很奇怪，像是劣质奶粉冲的，一股怪味",
        "茶底很涩，喝完嗓子不舒服，差评",
        "包装上没有标注口味，拿错了也不知道，麻烦",
        "价格比别家贵很多，但味道却不如别家，不划算",
        "珍珠没煮熟，中间是硬的，根本没法吃",
        "水果茶里的水果是罐头的，不是新鲜的，欺骗消费者",
        "热饮太烫了，根本没法立刻喝，只能等凉",
        "半糖还是很甜，怀疑糖不要钱，喝着不舒服",
        "奶盖太咸了，和茶底一点都不搭，很难喝",
        "茶底种类太少，只有一种选择，没新意",
        "小料是机器做的，口感很差，不如手工的",
        "大杯和中杯差别不大，感觉被骗了",
        "冷藏后味道更差，一股腥味，太难喝了",
        "控糖选项只有一种，不够灵活，不人性化",
        "新品出得快但质量差，都是为了圈钱",
        "排队久就算了，味道还不好，太失望了",
        "外卖包装和店里不一样，偷工减料，差评",
        "茶香太淡，奶味太浓，腻得不行，不好喝",
        "珍珠放久了，太软了没嚼劲，口感差",
        "水果茶太酸了，像是没熟的水果，难喝"
    ]

    # 合并数据并打乱顺序
    texts = positive + negative
    labels = [1] * 50 + [0] * 50  # 1:正面, 0:负面
    # 打乱顺序
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    return texts, labels


# --------------------------
# 2. 模型构建（简化）
# --------------------------
class SimpleLSTMTextClassifier(nn.Layer):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=32, num_classes=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        # 单层LSTM降低复杂度
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,  # 简化为1层
            direction='bidirectional',
            dropout=0
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_embed = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x_embed = x_embed.transpose([1, 0, 2])  # [seq_len, batch_size, embedding_dim]
        lstm_out, _ = self.lstm(x_embed)  # [seq_len, batch_size, hidden_dim*2]
        last_out = lstm_out[-1, :, :]  # 取最后一个时间步
        last_out = self.dropout(last_out)
        logits = self.fc(last_out)
        return logits


# --------------------------
# 3. 模型训练（优化策略）
# --------------------------
def train_model(model, train_loader, val_loader, epochs=30, lr=0.001):
    # 学习率衰减策略
    scheduler = paddle.optimizer.lr.LinearWarmup(
        learning_rate=lr,
        warmup_steps=3,
        start_lr=0.0001,
        end_lr=lr
    )
    optimizer = paddle.optimizer.Adam(
        learning_rate=scheduler,
        parameters=model.parameters()
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for batch in train_loader:
            texts, labels = batch
            texts = paddle.to_tensor(texts)
            labels = paddle.to_tensor(labels)

            logits = model(texts)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            train_loss += loss.item()
            preds = paddle.argmax(logits, axis=1).numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.numpy())

        train_acc = accuracy_score(train_labels, train_preds)
        scheduler.step()  # 学习率衰减

        # 验证
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []
        with paddle.no_grad():
            for batch in val_loader:
                texts, labels = batch
                texts = paddle.to_tensor(texts)
                labels = paddle.to_tensor(labels)
                logits = model(texts)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = paddle.argmax(logits, axis=1).numpy()
                val_preds.extend(preds)
                val_true.extend(labels.numpy())
        val_acc = accuracy_score(val_true, val_preds)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss / len(train_loader):.4f}, Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss / len(val_loader):.4f}, Acc: {val_acc:.4f}")
        print("-" * 50)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            paddle.save(model.state_dict(), "best_model.pdparams")
            print(f"Saved best model with val acc: {best_val_acc:.4f}")

    return model


# --------------------------
# 4. 模型评价与预测（保持不变）
# --------------------------
def evaluate_model(model, test_loader, id2label):
    model.eval()
    all_preds = []
    all_true = []
    with paddle.no_grad():
        for batch in test_loader:
            texts, labels = batch
            texts = paddle.to_tensor(texts)
            logits = model(texts)
            preds = paddle.argmax(logits, axis=1).numpy()
            all_preds.extend(preds)
            all_true.extend(labels.numpy())
    all_preds_labels = [id2label[p] for p in all_preds]
    all_true_labels = [id2label[t] for t in all_true]
    acc = accuracy_score(all_true_labels, all_preds_labels)
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_true_labels, all_preds_labels, zero_division=0))  # 解决警告
    return acc


def predict_text(model, processor, text):
    model.eval()
    with paddle.no_grad():
        text_ids = processor.text_to_ids(text)
        text_tensor = paddle.to_tensor([text_ids])
        logits = model(text_tensor)
        pred_id = paddle.argmax(logits, axis=1).item()
        pred_label = processor.id2label[pred_id]
        probs = paddle.nn.functional.softmax(logits, axis=1).numpy()[0]
        pred_prob = probs[pred_id]
        return pred_label, pred_prob


# --------------------------
# 主函数
# --------------------------
def main():
    # 1. 加载数据（扩展后）
    print("加载数据...")
    texts, labels = load_sample_data()
    print(f"数据集规模：{len(texts)}条（正负各50条）")

    # 2. 预处理
    print("预处理数据...")
    processor = TextProcessor(max_vocab_size=1000, max_seq_len=50)
    processor.build_vocab(texts)
    X, y = processor.process_dataset(texts, labels)

    # 划分数据集（比例调整为7:1.5:1.5）
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=100)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=100)

    # 数据加载器（增大batch_size）
    batch_size = 8
    train_dataset = paddle.io.TensorDataset([paddle.to_tensor(X_train), paddle.to_tensor(y_train)])
    val_dataset = paddle.io.TensorDataset([paddle.to_tensor(X_val), paddle.to_tensor(y_val)])
    test_dataset = paddle.io.TensorDataset([paddle.to_tensor(X_test), paddle.to_tensor(y_test)])
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = paddle.io.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = paddle.io.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3. 构建模型（简化版）
    print("构建模型...")
    vocab_size = len(processor.word2id)
    model = SimpleLSTMTextClassifier(
        vocab_size=vocab_size,
        embedding_dim=64,
        hidden_dim=32,
        num_classes=2,
        dropout=0.3
    )

    # 4. 训练（增加轮次）
    print("开始训练...")
    model = train_model(model, train_loader, val_loader, epochs=30, lr=0.001)

    # 加载最佳模型
    model.set_state_dict(paddle.load("best_model.pdparams"))

    # 5. 评估
    print("评估模型...")
    evaluate_model(model, test_loader, processor.id2label)

    # 6. 预测示例
    print("\n预测示例...")
    test_samples = [
        "这款奶茶超赞，奶盖浓郁，茶香清新，性价比超高",
        "难喝到爆，甜度过高，珍珠不新鲜，再也不买了",
        "味道中规中矩，茶味有点淡，小料给的还挺足"
    ]
    for sample in test_samples:
        label, prob = predict_text(model, processor, sample)
        print(f"文本: {sample}")
        print(f"预测类别: {'正面' if label == 1 else '负面'}, 概率: {prob:.4f}\n")


if __name__ == "__main__":
    main()