import scipy.io as scio
import numpy as np


def mat2csv(inputFile, outputFile):
    """
    读取 mat 文件，提取数据和标签转存为 csv 文件
    :param inputFile: 要读取的 mat 文件完整路径
    :param outputFile: 要存储的 csv 文件路径
    :return: None
    """
    matDict = scio.loadmat(inputFile)  # 读取 mat 文件，字典形式
    emg = matDict['emg']  # emg 信号
    label = matDict['restimulus']  # 标签列表

    index = []  # 存放活动信号的位置索引
    for i in range(len(label)):
        if (label[i] != 0) and (label[i] <= 8):
            index.append(i)

    actEmg = emg[index, :]  # 活动EMG
    actLabel = label[index, :] - 1  # 标签值从0开始

    matrix = np.hstack((actEmg, actLabel))
    np.savetxt(outputFile, matrix, delimiter=',')
    print('文件已存储', matrix.shape)
