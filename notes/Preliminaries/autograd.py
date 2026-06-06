import torch
x = torch.arange(4.0)
# print(x)
x.requires_grad_(True)
y = 2 * torch.dot(x, x)
# print(y)
y.backward()
# print(x.grad)       # gradient of x to y
x.grad.zero_()      # default to sum up
y = x.sum()
y.backward()
print(x.grad)