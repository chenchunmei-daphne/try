# confusion_matrix.py
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



cm = np.array([[5311, 560], [53, 3616]])

plt.plot(cm[0,:])
plt.show()
exit()
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred Negative', 'Pred Positive'],
            yticklabels=['True Negative', 'True Positive'])
plt.title('Confusion Matrix')
plt.tight_layout()

# 尝试显示
plt.show(block=True)  # block=True 保持窗口打开