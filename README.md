# sEMG

NinaPro数据下载链接：https://ninapro.hevs.ch/index.html

- S1_E1_A1.mat 对应 Exercise B
- S1_E2_A1.mat 对应 Exercise C
- S1_E3_A1.mat 对应 Exercise D

预处理（参考https://github.com/malele4th/sEMG_DeepLearning）：

1. 运行 getData.py 得到要用的原始数据（也可以调用getDataFunc.py函数接口）
2. 运行 featureUtils.py 进行数据滑窗分割

# domainbed

download.py可用于下载并提取常用公开数据集，目前支持PACS数据集。

