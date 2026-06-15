import torch
from torch import nn
from d2l import torch as d2l


# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# transform = transforms.Compose([
#     transforms.Resize(96),
#     transforms.ToTensor()
# ])

# train_data = datasets.FashionMNIST(
#     root="../data",
#     train=True,
#     transform=transform,
#     download=True
# )

# test_data = datasets.FashionMNIST(
#     root="../data",
#     train=False,
#     transform=transform,
#     download=True
# )

# train_iter = DataLoader(
#     train_data,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=0
# )

# test_iter = DataLoader(
#     test_data,
#     batch_size=batch_size,
#     shuffle=False,
#     num_workers=0
# )



# def evaluate_accuracy_gpu(net, data_iter, device = try_mps()): #@save
#     """使用GPU计算模型在数据集上的精度"""
#     if isinstance(net, nn.Module):
#         net.eval()  # 设置为评估模式
#         if not device:
#             device = next(iter(net.parameters())).device
#     # 正确预测的数量，总预测的数量
#     metric = d2l.Accumulator(2)
#     with torch.no_grad():
#         for X, y in data_iter:
#             if isinstance(X, list):
#                 # BERT微调所需的（之后将介绍）
#                 X = [x.to(device) for x in X]
#             else:
#                 X = X.to(device)
#             y = y.to(device)
#             metric.add(d2l.accuracy(net(X), y), y.numel())
#     return metric[0] / metric[1]

# def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
#     """用GPU训练模型(在第六章定义)"""
#     def init_weights(m):
#         if type(m) == nn.Linear or type(m) == nn.Conv2d:
#             nn.init.xavier_uniform_(m.weight)
#     net.apply(init_weights)
#     print('training on', device)
#     net.to(device)
#     optimizer = torch.optim.SGD(net.parameters(), lr=lr)
#     loss = nn.CrossEntropyLoss()
#     timer, num_batches = d2l.Timer(), len(train_iter)
#     for epoch in range(num_epochs):
#         # 训练损失之和，训练准确率之和，样本数
#         metric = d2l.Accumulator(3)
#         net.train()
#         for i, (X, y) in enumerate(train_iter):
#             print("epoch: ", epoch, "/ i: ", i)
#             timer.start()
#             optimizer.zero_grad()
#             X, y = X.to(device), y.to(device)
#             y_hat = net(X)
#             l = loss(y_hat, y)
#             l.backward()
#             optimizer.step()
#             with torch.no_grad():
#                 metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
#             timer.stop()
#             train_l = metric[0] / metric[2]
#             train_acc = metric[1] / metric[2]
#         test_acc = evaluate_accuracy_gpu(net, test_iter, device)

#     print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
#           f'test acc {test_acc:.3f}')
#     print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
#           f'on {str(device)}')


# def try_mps():
#     if torch.backends.mps.is_available():
#         return torch.device("mps")
#     return torch.device("cpu")
