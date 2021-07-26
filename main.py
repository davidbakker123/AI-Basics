import numpy as np
import matplotlib.pyplot as plt

input_size = 2
output_size = 1
layers = 20
nodes_per_layer = 10
batch_size = 1000

f = lambda x: np.where(x < 0, 0, x) 

grid = np.zeros(shape = [batch_size, batch_size, 2])
for j in range(batch_size):
    for i in range(batch_size):
        grid[j, i, 0] = j
        grid[j, i, 1] = i

### Initialize weights and biases ###
w = []
b = []

w.append(np.random.normal(0, 1, size = [input_size, nodes_per_layer]))
b.append(np.random.normal(0, 1, size = [batch_size, nodes_per_layer]))
for i in range(layers - 2):
    w.append(np.random.normal(0, 1, size = [nodes_per_layer, nodes_per_layer]))
    b.append(np.random.normal(0, 1, size = [batch_size, nodes_per_layer]))
w.append(np.random.normal(0, 1, size = [nodes_per_layer, output_size]))
b.append(np.random.normal(0, 1, size = [batch_size, output_size]))

### Go through network ###
z = np.zeros(shape = [batch_size, batch_size])

for j in range(batch_size):
    input_vec = grid[j, :]
    for i in range(layers - 1):
        out = np.matmul(input_vec, w[i]) + b[i]
        out = f(out)
        input_vec = out
    out = np.matmul(input_vec, w[-1]) + b[-1]
    z[:, j] = np.transpose(out)

plt.imshow(z)
plt.show()