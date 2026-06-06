import torch

A = torch.arange(20).reshape(4,5)
# print(A, "\n", A.T)
B = A.clone()
print(A.size())
print(A + B)