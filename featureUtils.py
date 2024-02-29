import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

file_path = 's1_raw.csv'  # 数据路径
dataFrame = pd.read_csv(file_path, header=None)  # 读取数据
emg = dataFrame.iloc[:, :-1].values  # 转换为array类型
label = dataFrame.iloc[:, -1].values  # 转换为array类型
emg = emg.astype(np.float32)  # 浮点型
label = label.astype(np.int32)  # 整型

emg = emg * 20000

featureData = []  # 存储样本数据
featureLabel = []  # 存储样本标签

classes = 8  # 八类静态手势
timeWindow = 200  # 窗口长度
strideWindow = 100  # 步长

for i in range(classes):  # 遍历八种手势标签
    index = []  # 存储单个手势的索引值
    for j in range(label.shape[0]):  # 遍历标签列表
        if label[j] == i:  # 标签列表元素 == 手势标签
            index.append(j)
    iemg = emg[index, :]  # 单个手势的数据
    length = math.floor((iemg.shape[0] - timeWindow) / strideWindow)  # 单个手势样本数量
    print("class ", i, ",number of sample: ", iemg.shape[0], length)

    for j in range(length):
        example = iemg[strideWindow * j:strideWindow * j + timeWindow, :]  # 样本划分
        featureData.append(example)  # 添加样本
        featureLabel.append(i)       # 添加标签

featureData = np.array(featureData)
featureLabel = np.array(featureLabel)
print(featureData.shape)
print(featureLabel.shape)
featureData = featureData.reshape(featureData.shape[0], -1)
print(featureData.shape)
np.savetxt('s1_featureData.csv', featureData, delimiter=',')
np.savetxt('s1_featureLabel.csv', featureLabel, delimiter=',')