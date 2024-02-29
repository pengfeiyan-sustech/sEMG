import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

E1 = scio.loadmat('../dataset/NinaproDB2/DB2_s1/S1_E1_A1.mat')      # Exercise B    取前八种手势即可
E2 = scio.loadmat('../dataset/NinaproDB2/DB2_s1/S1_E2_A1.mat')      # Exercise C
E3 = scio.loadmat('../dataset/NinaproDB2/DB2_s1/S1_E3_A1.mat')      # Exercise D

print(E1.keys())
print(E2.keys())
print(E3.keys())

E1_emg = E1['emg']
E2_emg = E2['emg']
E3_emg = E3['emg']

E1_label = E1['restimulus']
E2_label = E2['restimulus']
E3_label = E3['restimulus']

index1 = []
for i in range(len(E1_label)):
    if E1_label[i] != 0 and E1_label[i] <= 8:
        index1.append(i)
label1 = E1_label[index1, :]
emg1 = E1_emg[index1, :]

index2 = []
for i in range(len(E2_label)):
    if E2_label[i] != 0:
        index2.append(i)
label2 = E2_label[index2, :]
emg2 = E2_emg[index2, :]

index3 = []
for i in range(len(E3_label)):
    if E3_label[i] != 0:
        index3.append(i)
label3 = E3_label[index3, :]
emg3 = E3_emg[index3, :]

plt.plot(label1)
plt.plot(emg1[:, 5]*20000)
# plt.plot(E1_emg[0:200000, 5]*20000)
# plt.plot(E1_label[0:200000])

plt.show()