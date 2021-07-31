import numpy as np
import matplotlib.pyplot as plt
import time

input_size = 2
output_size = 1
layers = 10
nodes_per_layer = [2] + (layers - 2) * [10] + [1]
batch_size = 100

f =  lambda x: 1 / (1 + np.exp(-x)) # #f = lambda x: x * np.tanh(np.log(1 + np.exp(x))) # np.where(x < 0, 0.0, x)  #
#f_prime = lambda x: np.where(x < 0, 0.0, 1.0)
f_prime = lambda x: np.exp(-x)/(1+np.exp(-x))**2

### Initialize weights and biases ###
min, max = -1, 1
distr = lambda x: np.random.uniform(0, 1, size = x)
w = [distr([nodes_per_layer[i], nodes_per_layer[i+1]]) for i in range(layers - 1) ]
b = [distr([1, nodes_per_layer[i+1]]) for i in range(layers - 1) ]

### Go through network ###
y_layer = [ np.zeros(nodes_per_layer[i]) for i in range(layers) ] 
df_layer = [ np.zeros(nodes_per_layer[i+1]) for i in range(layers - 1) ]
dw_layer = [ np.zeros([nodes_per_layer[i], nodes_per_layer[i+1]]) for i in range(layers - 1) ]
db_layer = [ np.zeros([1, nodes_per_layer[i+1]]) for i in range(layers - 1) ]

def apply_network(input_vec):
    y_layer[0] = input_vec
    y = input_vec
    for i in range(layers - 1):
        z = np.matmul(y, w[i]) + b[i]
        y_layer[i + 1] = f(z)
        df_layer[i] = f_prime(z)
        y = y_layer[i + 1]
    return y

def backprop(output, y_target):
    output = y_layer
    Delta = (y_layer[-1] - y_target) * df_layer[-1] #  (y-y*) * df/dz   # y_layer[-1].shape = (1,1)
    dw_layer[-1] = np.matmul(np.transpose(y_layer[-2]), Delta) / batch_size 
    db_layer[-1] = np.array([Delta.sum(axis = 0) / batch_size])
    for i in range(layers - 2):
        Delta = np.matmul(Delta, np.transpose(w[-1-i])) * df_layer[-2-i] # Delta_new,k = sum_j Delta_old,j * w_jk * f'(z_k)
        dw_layer[-2-i] = np.matmul(np.transpose(y_layer[-3-i]), Delta) / batch_size
        db_layer[-2-i] = np.array([Delta.sum(axis = 0) / batch_size]) / batch_size

def gradient_descent(delta):
    for i in range(layers - 1):   
        w[i] -= delta * dw_layer[i]
        b[i] -= delta * db_layer[i]        

def learning_step(target, delta):
    #output = apply_network(input)
    target = np.transpose(np.array([target]))
    backprop(y_layer, target)
    gradient_descent(delta)
    cost = (1/2) * np.mean((y_layer[-1]-target).sum(1)**2)
    return cost

def target_func(x, y):
    return (np.sin(x*3) + np.sin(y*3))

x = np.linspace(0, 1, batch_size)
y = np.linspace(0, 1, batch_size)
xx, yy = np.meshgrid(x, y)
line = np.ndarray(shape = [batch_size, input_size])

steps = 500
cost_arr = np.zeros(steps)
z = np.zeros(shape = [batch_size, batch_size])
for k in range(steps):
    print(k)
    for j in range(batch_size):
        line[:, 0], line[:, 1] = xx[j, :], yy[j, :]
        output_vec = apply_network(line)
        z[:, j] = np.transpose(output_vec)
        cost = learning_step(target_func(xx[j, :], yy[j, :]), 0.01)
    cost_arr[k] = cost
    
plt.plot(cost_arr)
plt.ylabel("Cost")
plt.xlabel("steps")
plt.show()

plt.figure(1) 
plt.imshow(z)
plt.show()

plt.figure(2)
plt.imshow(target_func(xx, yy))
plt.show()
"""
plt.ion()
fig = plt.figure()
axis = fig.add_subplot(111)
for i in range(100):
    axis.plot(i, i, 'o')
    plt.draw()
    time.sleep(1)
"""