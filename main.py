import numpy as np
import matplotlib.pyplot as plt

input_size = 2
output_size = 1
layers = 50 # Number of layers (without input layer)
nodes_per_layer = [2] + (layers - 1) * [50] + [1]
batch_size = 100

f = lambda x: np.where(x < 0, 0, x)  # 1 / (1 + np.exp(-x)) # #f = lambda x: x * np.tanh(np.log(1 + np.exp(x)))
f_prime = lambda x: np.where(x < 0, 0.0, 1.0)

grid = np.zeros(shape = [batch_size, batch_size, 2])
for j in range(batch_size):
    for i in range(batch_size):
        grid[j, i, 0], grid[j, i, 1] = j, i
grid /= batch_size

### Initialize weights and biases ###
min, max = -1, 1
distr = lambda x: np.random.normal(0, 1, size = x)
w = [distr([nodes_per_layer[i], nodes_per_layer[i+1]]) for i in range(layers) ]
b = [distr([1, nodes_per_layer[i+1]]) for i in range(layers) ]

### Go through network ###
y_layer = [ np.zeros(nodes_per_layer[i]) for i in range(layers + 1) ] 
df_layer = [ np.zeros(nodes_per_layer[i+1]) for i in range(layers) ]
dw_layer = [0] * layers
db_layer = [0] * layers

def apply_network(input_vec):
    y_layer[0] = input_vec
    y = input_vec
    for i in range(layers):
        z = np.matmul(y, w[i]) + b[i]
        y_layer[i + 1] = f(z)
        df_layer[i] = f_prime(z)
        y = y_layer[i + 1]
    return y

def backprop(y_target):
    Delta = (y_layer[-1] - y_target) * df_layer[-1] #  (y-y*) * df/dz   # y_layer[-1].shape = (1,1)
    dw_layer[-1] = np.matmul(np.transpose(y_layer[-2]), Delta) / batch_size 
    db_layer[-1] = Delta.sum(axis = 0) / batch_size
    for i in range(layers - 1):
        Delta = np.matmul(Delta, np.transpose(w[-1-i])) * df_layer[-2-i] # Delta_new,k = sum_j Delta_old,j * w_jk * f'(z_k)
        dw_layer[-2-i] = np.matmul(np.transpose(y_layer[-3-j]), Delta) / batch_size
        db_layer[-2-i] = Delta.sum(axis = 0) / batch_size

z = np.zeros(shape = [batch_size, batch_size])
for j in range(batch_size):
    input_vec = grid[j, :]
    output_vec = apply_network(input_vec)
    z[:, j] = np.transpose(output_vec)  

#backprop(np.ones(shape = [len(y_layer[-1]) , 1]))
plt.imshow(z)
plt.show()