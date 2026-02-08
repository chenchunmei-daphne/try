# import matplotlib
# matplotlib.use('TkAgg') # 更改后端为TkAgg
# import matplotlib.pyplot as plt
# x = [1, 2, 3, 4]
# y = [10, 20, 25, 30]
# plt.plot(x, y)
# plt.show()

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

plt.ioff()

fig, ax = plt.subplots()     # ★ 显式创建 Figure 和 Axes
ax.plot([1, 2, 3], [1, 4, 9])

fig.canvas.draw()            # ★ 关键：强制绘制
plt.show(block=True)

input("Press Enter to exit")
