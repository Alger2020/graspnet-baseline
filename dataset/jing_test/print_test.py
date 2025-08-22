import numpy as np
from PIL import Image

# 方法1：通过PIL读取并转换
image_pil = Image.open('/T13/jing/graspnet-baseline/dataset/graspnet/scenes/scene_0000/kinect/label/0000.png')
image_array = np.array(image_pil)  # 转换为NumPy数组
print("形状（PIL）:", image_array.shape)
# 打印最大值和最小值
print("最大值:", np.max(image_array))
print("最小值:", np.min(image_array))
# 打印所有出现过的唯一值列表（如果数据量不大）
unique_values = np.unique(image_array)
print("出现过的数字列表:", unique_values)

# # 方法2：通过matplotlib读取
# import matplotlib.pyplot as plt
# image_plt = plt.imread('example.png')  # 直接读取为NumPy数组
# print("形状（Matplotlib）:", image_plt.shape)

# # 方法3：通过OpenCV读取（需注意BGR顺序）
# import cv2
# image_cv2 = cv2.imread('example.png')  # 默认BGR通道
# print("形状（OpenCV）:", image_cv2.shape)
import scipy.io
import numpy as np

# 读取 .mat 文件
mat_data = scipy.io.loadmat('/T13/jing/graspnet-baseline/dataset/graspnet/scenes/scene_0000/kinect/meta/0000.mat')  # 替换为你的文件路径

# 显示文件中的所有变量名（跳过内部变量如 '__header__'）
print("--- 变量列表 ---")
for key in mat_data.keys():
    if not key.startswith('__'):
        print(key)

# 提取变量（假设变量名为 'data'）
variable_name = 'factor_depth'  # 替换为实际变量名
data = mat_data[variable_name]

# 显示变量信息
print(f"\n--- 变量 '{variable_name}' 的详细信息 ---")
print("数据类型:", type(data))
print("数组维度:", data.ndim)
print("形状（shape）:", data.shape)
print("数据类型（dtype）:", data.dtype)
print("\n前5行数据:\n", data[:5] if data.ndim > 0 else data)

