import numpy as np
import matplotlib.pyplot as plt

batch_size = 1000
nodes_per_layer = [2] + 10 * [100] + [1]
layers =  len(nodes_per_layer) - 1 # excluding input layer
input_size = nodes_per_layer[0]
output_size = nodes_per_layer[-1]

w = [np.random.uniform(low = -0.1, high = 0.1, size = [nodes_per_layer[i], nodes_per_layer[i + 1]]) for i in range(layers)]
b = [np.zeros(nodes_per_layer[i + 1]) for i in range(layers)]
#b = [np.random.uniform(low = -1, high = 1, size = [1, nodes_per_layer[i + 1]]) for i in range(layers)]

dw_layer = [np.zeros([nodes_per_layer[i], nodes_per_layer[i + 1]]) for i in range(layers)]
db_layer = [np.zeros(nodes_per_layer[i+1]) for i in range(layers)]

y_layer = [np.zeros(nodes_per_layer[i]) for i in range(layers + 1)]
df_layer = [np.zeros(nodes_per_layer[i + 1]) for i in range(layers)]

act = lambda z:  1 / (1 + np.exp(-z)) # np.where(z < 0, 0, z) #
d_act = lambda z:  np.exp(-z) * (1 / (1 + np.exp(-z)))**2 # np.where(z < 0, 0, 1) #

print(w[0])

def apply_network(y_in):
    global y_layer, df_layer, w, b
    y_layer[0] = y_in
    for i in range(layers):
        z = np.matmul(y_layer[i], w[i]) + b[i]
        y_layer[i+1] = act(z)
        df_layer[i] = d_act(z)   
    return (y_layer[-1])

def backprop(y_target):
    global y_layer, df_layer, dw_layer, db_layer, w, b
    Delta = (y_layer[-1] - y_target) * df_layer[-1]
    dw_layer[-1] = np.matmul(np.transpose(y_layer[-2]), Delta) / batch_size
    db_layer[-1] = Delta.sum(0) / batch_size
    for i in range(layers - 1):
        Delta = np.matmul(Delta, np.transpose(w[-1-i])) * df_layer[-2-i]   
        dw_layer[-2-i] = np.matmul(np.transpose(y_layer[-3-i]), Delta) #/ batch_size
        db_layer[-2-i] = Delta.sum(0) / batch_size

def gradient_descent(eta):
    global w, b, dw_layer, db_layer
    #for i in dw_layer:
        #print(i)
    for i in range(layers):
        w[i] -= eta * dw_layer[i]
        b[i] -= eta * db_layer[i]

def learning_step(y_in, y_target, eta):
    y_out = apply_network(y_in)
    backprop(y_target)
    gradient_descent(eta)
    cost = ((y_target - y_out)**2).sum() / batch_size
    return (cost)

def make_batch(batch_size):
    y_in = np.random.uniform(-0.5, 0.5, size = [batch_size, 2])
    y_target = np.zeros([batch_size, 1])
    y_target[:, 0] = target_function(y_in[:, 0], y_in[:, 1])
    return (y_in, y_target)
    
def target_function(x, y):
    r2=x**2+y**2
    return(np.exp(-5*r2)*abs(x+y))
    #return np.sin(3 * x) + np.sin(3 * y)

eta = 0.01
steps = 100
cost = np.zeros(steps)
for i in range(steps):
    y_in, y_target = make_batch(batch_size)
    print(i)
    cost[i] = learning_step(y_in, y_target, eta)

plt.figure(0)
plt.plot(cost)
plt.show()

print(w[0])

#x = np.linspace(-0.5, 0.5, batch_size)
#X0, X1 = np.meshgrid(x, x)

#z = np.zeros(shape = [batch_size, batch_size])
#input = np.zeros(shape = [batch_size, 2])
#for i in range(batch_size):
    #input[:, 0] = X0[i, :]
    #input[:, 1] = X1[i, :]
    #output = apply_network(input)
    #z[:, i] = np.transpose(output)

#plt.figure(1)
#plt.imshow(z)
#plt.show()