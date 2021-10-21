import numpy as np
import pandas as pd

x20 = pd.read_excel('tb_data.xlsm', sheet_name=2)[5:].to_numpy()
x16 = pd.read_excel('tb_data.xlsm', sheet_name=3)[5:].to_numpy()
x12 = pd.read_excel('tb_data.xlsm', sheet_name=4)[5:].to_numpy()
x9 = pd.read_excel('tb_data.xlsm', sheet_name=5)[5:].to_numpy()
x6 = pd.read_excel('tb_data.xlsm', sheet_name=6)[5:].to_numpy()

data = np.array([x6, x9, x12, x16, x20], dtype=object)
print(data.shape)
print(data[0][0, 0], data[0][-1, 0])
print(data[1][0, 0], data[1][-1, 0])
print(data[2][0, 0], data[2][-1, 0])
print(data[3][0, 0], data[3][-1, 0])
print(data[4][0, 0], data[4][-1, 0])

np.save('pdds.npy', data)