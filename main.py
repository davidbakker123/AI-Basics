import numpy as np
import matplotlib.pyplot as plt

input_size = 2
output_size = 1
layers = 10
nodes_per_layer = [2] + (layers - 1) * [10] + [1]
batch_size = 1000

print(nodes_per_layer, layers)

f = lambda x:  1 / (1 + np.exp(-x)) # #f = lambda x: x * np.tanh(np.log(1 + np.exp(x))) # np.where(x < 0, 0.0 , x) 
#f_prime = lambda x: np.where(x < 0, 0.0, 1.0)
f_prime = lambda x: np.exp(-x)/(1+np.exp(-x))**2

### Initialize weights and biases ##
min, max = -1, 1
distr = lambda x: np.random.uniform(-1, 1, size = x)
w = [distr([nodes_per_layer[i], nodes_per_layer[i+1]]) for i in range(layers) ]
b = [distr(nodes_per_layer[i+1]) for i in range(layers) ]

### Go through network ###
y_layer = [ np.zeros(nodes_per_layer[i]) for i in range(layers + 1) ] 
df_layer = [ np.zeros(nodes_per_layer[i+1]) for i in range(layers) ]
dw_layer = [ np.zeros([nodes_per_layer[i], nodes_per_layer[i+1]]) for i in range(layers) ]
db_layer = [ np.zeros(nodes_per_layer[i+1]) for i in range(layers) ]

def net_f_df(z):
    val = 1/(1 + np.exp(-z))
    return (val, np.exp(-z) * val**2)

def forward_step(y, w, b):
    z = np.dot(y,w) + b
    return (net_f_df(z))

def apply_network(input_vec):
    global y_layer, w, b, df_layer, layers, batch_size
    y_layer[0] = input_vec
    y = input_vec
    for i in range(layers):
        y, df = forward_step(y, w[i], b[i])
        df_layer[i] = df
        y_layer[i + 1] = y
    return (y)

def apply_net_simple(y_in):
    global w, b, layers, batch_size
    y=y_in 
    y_layer[0]=y
    for j in range(layers):
        y,df=forward_step(y,w[j],b[j])
    return(y)

def backward_step(delta, w, df):
    return (np.dot(delta, np.transpose(w)) * df)

def backprop(y_target):
    global y_layer, df_layer, w, b
    global dw_layer, db_layer, layers, batch_size
    delta = (y_layer[-1] - y_target) * df_layer[-1] #  (y-y*) * df/dz   # y_layer[-1].shape = (1,1)
    dw_layer[-1] = np.dot(np.transpose(y_layer[-2]), delta) / batch_size 
    db_layer[-1] = delta.sum(0) / batch_size
    for i in range(layers - 1):
        delta = backward_step(delta, w[-1-i], df_layer[-2-i]) #np.matmul(Delta, np.transpose(w[-1-i])) * df_layer[-2-i] # Delta_new,k = sum_j Delta_old,j * w_jk * f'(z_k)
        dw_layer[-2-i] = np.dot(np.transpose(y_layer[-3-i]), delta)/ batch_size
        db_layer[-2-i] = delta.sum(0)/batch_size


def gradient_descent(delta):
    global w, b, dw_layer, db_layer, layers     
    for i in range(layers):   
        w[i] -= delta * dw_layer[i]
        b[i] -= delta * db_layer[i]        

def learning_step(inputs, targets, delta):
    global batch_size
    y_out = apply_network(inputs)
    backprop(targets)
    gradient_descent(delta)
    cost = ((targets- y_out)**2).sum()/batch_size
    return (cost)

#def target_func(x, y):
    #return np.sin(3*x) + np.sin(3*y)

def myFunc(x0,x1):
    r2=x0**2+x1**2
    return(np.exp(-5*r2)*np.abs(x1+x0))

def create_batch(batch_size):
    inputs = np.random.uniform(-0.5, 0.5, size = [batch_size, 2])
    targets = np.zeros(shape = [batch_size, 1])
    targets[:, 0] = myFunc(inputs[:, 0], inputs[:, 1])
    return (inputs, targets)

steps = 5000
costs = np.zeros(steps)
#for i in range(steps):
    #print(i)
    #inputs, targets = create_batch(batch_size)
    #costs[i] = learning_step(inputs, targets, 0.001)

#plt.figure(0)
#plt.plot(costs)
#plt.ylabel("Cost")
#plt.xlabel("steps")
#plt.show()

#plt.figure(1)
x = np.linspace(-0.5, 0.5, 40)
y = np.linspace(-0.5, 0.5, 40)
xx, yy = np.meshgrid(x, y)
#plt.imshow(myFunc(xx, yy),interpolation='nearest',origin='lower')
#plt.show()

test_batchsize = np.shape(xx)[0]*np.shape(xx)[1]
testsample = np.zeros([test_batchsize, 2])
testsample[:, 0] = xx.flatten()
testsample[:, 1] = yy.flatten()
testoutput = apply_net_simple(testsample)
#myim = plt.imshow(np.reshape(testoutput, np.shape(xx)), origin = 'lower', interpolation='none')

from IPython.display import clear_output
from time import sleep

eta=0.01 # learning rate
nsteps=1000

plt.ion()
fig,ax=plt.subplots(ncols=2,nrows=1,figsize=(8,4)) # prepare figure
ax[1].axis('off') # no axes
costs=np.zeros(nsteps)
for j in range(nsteps):
    clear_output(wait=True)
    
    # the crucial lines:
    y_in,y_target=create_batch(batch_size) # random samples (points in 2D)
    costs[j]=learning_step(y_in,y_target,eta) # train network (one step, on this batch)
    testoutput=apply_net_simple(testsample) # check the new network output in the plane
    
    ax[1].cla()
    img=ax[1].imshow(np.reshape(testoutput,np.shape(xx)),interpolation='nearest',origin='lower') # plot image
    ax[0].cla()
    ax[0].plot(costs)
    
    ax[0].set_title("Cost during training")
    ax[0].set_xlabel("number of batches")
    ax[1].set_title("Current network prediction")
    plt.show()
    plt.pause(0.01)
#plt.figure(2)
#print(w, b)

#plt.show()


"""
x = np.linspace(0, 1, batch_size)
y = np.linspace(0, 1, batch_size)
xx, yy = np.meshgrid(x, y)
line = np.ndarray(shape = [batch_size, input_size])


steps = 10
cost_arr = np.zeros(steps)
z = np.zeros(shape = [batch_size, batch_size])
for k in range(steps):
    for j in range(batch_size):
        line[:, 0], line[:, 1] = xx[j, :], yy[j, :]
        output_vec = apply_network(line)
        z[:, j] = np.transpose(output_vec)
        cost = learning_step(target_func(xx[j, :], yy[j, :]), 0.01)
    cost_arr[k] = cost
    print(k)   

create_batch(batch_size)
"""
"""
plt.plot(cost_arr)
plt.show()

plt.figure(1) 
plt.imshow(z)
plt.show()

plt.figure(2)
plt.imshow(target_func(xx, yy))
plt.show()
"""
"""
plt.ion()
fig = plt.figure()
axis = fig.add_subplot(111)
for i in range(100):
    axis.plot(i, i, 'o')
    plt.pause(1)
"""