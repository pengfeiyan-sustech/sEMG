import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

E1 = scio.loadmat('../dataset/NinaproDB2/DB2_s1/S1_E1_A1.mat')      # Exercise B    取前八种手势即可
print(E1.keys())
E1_emg = E1['emg']
E1_label = E1['restimulus']
index1 = []
for i in range(len(E1_label)):
    if E1_label[i] != 0 and E1_label[i] <= 8:
        index1.append(i)
label1 = E1_label[index1, :] - 1
emg1 = E1_emg[index1, :]
# plt.plot(emg1[:, 5]*20000)
# plt.plot(label1)
# plt.plot(E1_emg[0:200000, 5]*20000)
# plt.plot(E1_label[0:200000])
# plt.show()
print(emg1.shape)
print(label1.shape)
matrix = np.hstack((emg1, label1))

np.savetxt('s1_raw.csv', matrix, delimiter=',')

