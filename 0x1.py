import torch.nn as nn
import torch
from torch.nn.modules import loss 
import torch.optim as optim
import matplotlib.pyplot as plt

class Linear2(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.w = nn.Parameter(torch.randn(input_size, output_size) * 2) 
        self.b = nn.Parameter(torch.randn(output_size))

    def forward(self, x):
        return torch.matmul(x, self.w) * 3 + self.b * 2 + 100

class net(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(1,2), nn.ReLU()] + 100 * [nn.Linear(2, 2), nn.ReLU()] + [nn.Linear(2, 1)] 
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)

input = torch.randn(10, 2)
layer = Linear2(2, 1)
output = layer(input)

optimizer = optim.Adam(layer.parameters(), lr = 0.01)

loss_function = nn.MSELoss()

plt.ion()
y = []
x = []
for i in range(1000):
    optimizer.zero_grad()
    loss = loss_function(layer(input), torch.ones(10, 1))
    loss.backward()
    optimizer.step()

    x.append(i)
    y.append(loss.detach().item())
    plt.cla()
    plt.plot(x, y)
    plt.pause(0.01)
plt.show()


#net1 = net()
