import numpy as np
import matplotlib.pyplot as plt
import time

input_size = 2
output_size = 1
layers = 50
nodes_per_layer = 50
batch_size = 1000

f = lambda x: np.where(x < 0, 0, x)  #1 / (1 + np.exp(-x)) # 

grid = np.zeros(shape = [batch_size, batch_size, 2])
for j in range(batch_size):
    for i in range(batch_size):
        grid[j, i, 0], grid[j, i, 1] = j, i

grid /= batch_size
### Initialize weights and biases ###
w = []
b = []

min = -1
max = 1
uni = lambda x: np.random.uniform(min, max, size = x)

w.append(uni([input_size, nodes_per_layer]))
b.append(uni([1, nodes_per_layer]))
for i in range(layers - 2):
    w.append(uni([nodes_per_layer, nodes_per_layer]))
    b.append(uni([1, nodes_per_layer]))
w.append(uni([nodes_per_layer, output_size]))
b.append(uni([1, output_size]))

### Go through network ###
z = np.zeros(shape = [batch_size, batch_size])

for j in range(batch_size):
    input_vec = grid[j, :]
    for i in range(layers):
        out = f(np.matmul(input_vec, w[i]) + b[i])
        input_vec = out
    z[:, j] = np.transpose(out)

plt.imshow(z)
plt.show()