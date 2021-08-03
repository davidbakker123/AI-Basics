import numpy as np
import matplotlib.pyplot as plt

class Network:

    def __init__(self, nodes_per_layer, batch_size, eta, activation_function = "ReLU"):
        self.nodes_per_layer = nodes_per_layer
        self.batch_size = batch_size
        self.eta = eta
        self.layers =  len(nodes_per_layer) - 1 # excluding input layer
        self.input_size = nodes_per_layer[0]
        self.output_size = nodes_per_layer[-1]
        self.activation_function = activation_function

        self.w = [np.random.uniform(low = -0.1, high = 0.1, size = [self.nodes_per_layer[i], self.nodes_per_layer[i + 1]]) for i in range(self.layers)]
        self.b = [np.zeros(self.nodes_per_layer[i + 1]) for i in range(self.layers)]

        self.dw_layer = [np.zeros([self.nodes_per_layer[i], self.nodes_per_layer[i + 1]]) for i in range(self.layers)]
        self.db_layer = [np.zeros(self.nodes_per_layer[i+1]) for i in range(self.layers)]

        self.y_layer = [np.zeros(self.nodes_per_layer[i]) for i in range(self.layers + 1)]
        self.df_layer = [np.zeros(self.nodes_per_layer[i + 1]) for i in range(self.layers)]

    def act(self, z):
        if(self.activation_function == "ReLU"):
            return np.where(z < 0, 0, z) 
        if(self.activation_function == "sigmoid"):
            return 1 / (1 + np.exp(-z))
        else:
            print("Invalid activation function")
        
    def d_act(self, z):
        if(self.activation_function == "ReLU"):
            return np.where(z < 0, 0, 1) 
        if(self.activation_function == "sigmoid"):
            return np.exp(-z) * (1 / (1 + np.exp(-z)))**2 
        else:
            print("Invalid activation function")
            
    def apply_network(self, y_in):
        self.y_layer[0] = y_in
        for i in range(self.layers):
            z = np.matmul(self.y_layer[i], self.w[i]) + self.b[i]
            self.y_layer[i+1] = self.act(z)
            self.df_layer[i] = self.d_act(z)   
        return (self.y_layer[-1])

    def backprop(self, y_target):
        Delta = (self.y_layer[-1] - y_target) * self.df_layer[-1]
        self.dw_layer[-1] = np.matmul(np.transpose(self.y_layer[-2]), Delta) / self.batch_size
        self.db_layer[-1] = Delta.sum(0) / self.batch_size
        for i in range(self.layers - 1):
            Delta = np.matmul(Delta, np.transpose(self.w[-1-i])) * self.df_layer[-2-i]   
            self.dw_layer[-2-i] = np.matmul(np.transpose(self.y_layer[-3-i]), Delta) #/ self.batch_size
            self.db_layer[-2-i] = Delta.sum(0) / self.batch_size

    def gradient_descent(self):
        for i in range(self.layers):
            self.w[i] -= self.eta * self.dw_layer[i]
            self.b[i] -= self.eta * self.db_layer[i]

    def learning_step(self, y_in, y_target):
        y_out = self.apply_network(y_in)
        self.backprop(y_target)
        self.gradient_descent()
        cost = ((y_target - y_out)**2).sum() / self.batch_size
        return (cost)


def make_batch(batch_size):
    y_in = np.random.uniform(-0.5, 0.5, size = [batch_size, 2])
    y_target = np.zeros([batch_size, 1])
    y_target[:, 0] = target_function(y_in[:, 0], y_in[:, 1])
    return (y_in, y_target)
    
def target_function(x, y):
    #r2=x**2+y**2
    #return(np.exp(-5*r2)*abs(x+y))
    return (np.sin(10 * x) + np.sin(10 * y) + 1)

### Network setup ###
batch_size = 500
eta = 0.01
net = Network([2, 100, 100, 100, 100, 100, 1], batch_size = batch_size, eta = eta, activation_function = "ReLU")

### Learning loop ###
steps = 2000
cost = np.zeros(steps)
for i in range(steps):
    y_in, y_target = make_batch(batch_size)
    print(i)
    cost[i] = net.learning_step(y_in, y_target)

### Plotting stuff ###
x = np.linspace(-0.5, 0.5, batch_size)
X0, X1 = np.meshgrid(x, x)

z = np.zeros(shape = [batch_size, batch_size])
target = np.zeros(shape = [batch_size, batch_size])
input = np.zeros(shape = [batch_size, 2])
for i in range(batch_size):
    input[:, 0] = X0[i, :]
    input[:, 1] = X1[i, :]
    output = net.apply_network(input)
    z[:, i] = np.transpose(output)
    target[:, i] = np.transpose(target_function(input[:, 0], input[:, 1]))

plt.rcParams['figure.dpi']=100 # highres display
fig, ax = plt.subplots(1, 3, figsize = (15, 6))

ax[0].plot(cost)
ax[0].set_xlabel('steps')
ax[0].set_ylabel('Cost')
ax[1].imshow(z)
ax[1].axis("off")
ax[1].set_title("Network output")
ax[2].imshow(target)
ax[2].axis("off")
ax[2].set_title("Target function")
plt.show()