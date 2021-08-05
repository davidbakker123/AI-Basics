import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor()])
train_data = torchvision.datasets.MNIST(".", train = True, transform = transform, download = True)
val_data = torchvision.datasets.MNIST(".", train = False, transform = transform, download = True)

x, label = train_data[0]
print(x)
model = nn.Sequential()