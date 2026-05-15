import os
import numpy as np
import matplotlib.pyplot as plt

# 加载NPZ文件
# 设置路径
dataset_path = r"C:\Users\Administrator\Desktop\data1\data\zj1class0"
file_name = "class0_65.npz"
file_path = os.path.join(dataset_path, file_name)

# 读取 npz 文件
data = np.load(file_path)

# 获取数据
pos_data = data['pos']  # 正样本数据 (13564, 3)
neg_data = data['neg']  # 负样本数据 (13564, 3)

# 创建图像
fig, ax = plt.subplots(figsize=(8, 8))

# 隐藏所有边框
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_xticks([])
ax.set_yticks([])
    
# 设置图像边界和比例
ax.set_xlim(0, 256)
ax.set_ylim(0, 256)

# 绘制正样本点
ax.scatter(pos_data[:, 1], pos_data[:, 0], c='blue', marker='.', label='Positive', alpha=0.2, s=2)


# 绘制负样本点
ax.scatter(neg_data[:, 1], neg_data[:, 0], c='red', marker='.', label='Negative', alpha=0.2, s=2)

ax.invert_yaxis()

# # 添加网格线
# ax.grid(True, linestyle='--', alpha=0.5)

# # 设置图像标题和标签
# ax.set_xlabel('X Axis', fontsize=12)
# ax.set_ylabel('Y Axis', fontsize=12)

# # 添加图例
# ax.legend()

# 显示图像
plt.show()