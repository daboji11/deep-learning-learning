# import torch
# # x = torch.arange(12)
# # print(x)                # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
# # print(x.shape)          # torch.Size([12])
# # print(x.numel())        # number of element
# # x = x.reshape(3,4)
# # print(x)                # tensor([[ 0,  1,  2,  3], [ 4,  5,  6,  7], [ 8,  9, 10, 11]])

# x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
# y = torch.tensor([[2,1,4,3], [1,2,3,4], [4,3,2,1]])
# print(x == y)
# # print(torch.cat((x, y), dim=1))
# # print(x)

import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

import pandas as pd

data = pd.read_csv(data_file)
print(data)
input = data.iloc[:,0:2]
input = input.fillna(input.mean(numeric_only=True))
print(input)
input = pd.get_dummies(input, dummy_na=True)
print(input)
