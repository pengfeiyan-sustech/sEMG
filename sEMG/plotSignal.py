import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = './DB2Processed/subject1/s1_raw.csv'  # 数据路径
dataFrame = pd.read_csv(file_path, header=None)  # 读取数据
emg = dataFrame.iloc[:, :-1].values  # 转换为array类型
label = dataFrame.iloc[:, -1].values  # 转换为array类型
emg = emg.astype(np.float32)  # 浮点型
label = label.astype(np.int32)  # 整型

emg = emg * 20000

fig, axs = plt.subplots(12, 1)
color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
for i in range(12):
    axs[i].plot(emg[0:2000, i], color=color_list[0])
    # axs[i].set_title('Ch#' + str(i + 1))
    axs[i].get_xaxis().set_visible(False)
    axs[i].get_yaxis().set_visible(False)
    axs[i].axis('off')

plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=None, hspace=0.8)

plt.show()