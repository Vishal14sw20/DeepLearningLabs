# -*- coding: utf-8 -*-

import numpy as np


def analytic_weights(X, y):
    ## write your code here
    X_transpose = X.transpose()
    multiplication = X.dot(X_transpose)
    inverse = np.linalg.inv(multiplication)
    final = inverse.dot(X)
    w_star = final.dot(y)  # y is vector so we dont need to take transpose of it
    return w_star

    ## end of function


X = np.array([30,28,32,25,25,25,22,24,35,40])
y = np.array([25,30,27,40,42,40,50,45,30,25])
w_star = analytic_weights(X, y)

X.shape, y.shape, w_star.shape, w_star

"""

# Linear Regression Implementation from Scratch
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
from IPython import display
from matplotlib import pyplot as plt
import torch
import random

"""## Generating Data Sets
"""

num_inputs = 2
num_examples = 1000
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features = torch.zeros(size=(num_examples, num_inputs)).normal_()
labels = torch.matmul(features, true_w) + true_b
labels += torch.zeros(size=labels.shape).normal_(std=0.01)


def use_svg_display():
    # Display in vector graphics
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # Set the size of the graph to be plotted
    plt.rcParams['figure.figsize'] = figsize


set_figsize()
plt.figure(figsize=(10, 6))
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1);

"""The plotting function `plt` as well as the `use_svg_display` and `set_figsize` functions are defined in the `d2l` package. Now that you know how to make plots yourself, we will call `d2l.plt` directly for future plotting. To print the vector diagram and set its size, we only need to call `d2l.set_figsize()` before plotting, because `plt` is a global variable in the `d2l` package.


## Reading Data
"""


# This function has been saved in the d2l package for future use
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[j], labels[j]
        # The “take” function will then return the corresponding element based
        # on the indices


"""In general, note that we want to use reasonably sized minibatches to take advantage of the GPU hardware, which excels at parallelizing operations. Because each example can be fed through our models in parallel and the gradient of the loss function for each example can also be taken in parallel, GPUs allow us to process hundreds of examples in scarcely more time than it might take to process just a single example.

To build some intuition, let's read and print the first small batch of data examples. The shape of the features in each mini-batch tells us both the mini-batch size and the number of input features. Likewise, our mini-batch of labels will have a shape given by `batch_size`.
"""

batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    a = 0
    # print(X, y)
    break

"""
## Initialize Model Parameters
"""

w = torch.zeros(size=(num_inputs, 1)).normal_(std=0.01)
b = torch.zeros(size=(1,))
w, b

"""
Since nobody wants to compute gradients explicitly
(this is tedious and error prone),
we use automatic differentiation to compute the gradient.
See :numref:`chapter_autograd`
for more details.
Recall from the autograd chapter
that in order for `autograd` to know
that it should store a gradient for our parameters,
we need to invoke the `attach_grad` function,
allocating memory to store the gradients that we plan to take.
"""

w.requires_grad_(True)
b.requires_grad_(True)

"""## Define the Model
"""


# This function has been saved in the d2l package for future use
def linreg(X, w, b):
    return torch.matmul(X, w) + b


"""## Define the Loss Function
"""


# This function has been saved in the d2l package for future use
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


"""## Define the Optimization Algorithm
"""


# This function has been saved in the d2l package for future use
def sgd(params, lr, batch_size):
    for param in params:
        param.data.sub_(lr * param.grad / batch_size)
        param.grad.data.zero_()


"""## Training
"""

lr = 0.03  # Learning rate
num_epochs = 3  # Number of iterations
net = linreg  # Our fancy linear model
loss = squared_loss  # 0.5 (y-y')^2

for epoch in range(num_epochs):
    # Assuming the number of examples can be divided by the batch size, all
    # the examples in the training data set are used once in one epoch
    # iteration. The features and tags of mini-batch examples are given by X
    # and y respectively
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # Minibatch loss in X and y
        l.mean().backward()  # Compute gradient on l with respect to [w,b]
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print('epoch %d, loss %f' % (epoch + 1, train_l.mean().numpy()))

"""In this case, because we used synthetic data (that we synthesized ourselves!),
we know precisely what the true parameters are. Thus, we can evaluate our success in training by comparing the true parameters with those that we learned through our training loop. Indeed they turn out to be very close to each other.
"""

print('Error in estimating w', true_w - w.reshape(true_w.shape))
print('Error in estimating b', true_b - b)

"""
**Exercise:** Given a batch size b and a certain number of training examples n, how often are the parameters updated in each epoch (the number of epochs is num_epochs)? Compute the number of updates and return it from the following function.
"""


def compute_num_updates(num_epochs, n, b):
    num_updates = 0
    ## write your code here

    ## end of function
    return num_updates


"""**Exercise:** Deep learning is all about iterating and looking for the best loss for your application. Another loss function is the *L1* loss, which gives comparatively more importance to small errors and less for larger errors. Implement and return the *L1* loss, given `y` and `y_hat`. (Hint: assume both lists have the same shape)"""


def l1_loss(y, y_hat):
    a = 0

    # write your code here

    # end of function


"""# Concise Implementation of Linear Regression

The surge of deep learning has inspired the development of a variety of mature software frameworks, that
automate much of the repetitive work of implementing deep learning models. In the previous section we
relied only on NDarray for data storage and linear algebra and the auto-differentiation capabilities in the
autograd package. In practice, because many of the more abstract operations, e.g. data iterators, loss
functions, model architectures, and optimizers, are so common, deep learning libraries will give us library
functions for these as well.

We have used DataLoader to load the MNIST dataset in Section 4.5. In this section, we will learn how we can
implement the linear regression model in Section 5.2 much more concisely with DataLoader.

##  Generating Data Sets

To start, we will generate the same data set as that used in the previous section.
"""

import torch
import numpy as np


def synthetic_data(w, b, num_examples):
    """generate y = X w + b + noise"""
    X = np.random.normal(scale=1, size=(num_examples, len(w)))
    y = np.dot(X, w) + b
    y += np.random.normal(scale=0.01, size=y.shape)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float().reshape(-1, 1)
    return X, y


true_w = torch.Tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

"""##  Reading Data

Rather than rolling our own iterator, we can call upon DataLoader module to read data.
The first step will be to instantiate an ArrayDataset, which takes in one or more NDArrays as arguments. Here, we pass in features and
labels as arguments. Next, we will use the ArrayDataset to instantiate a DataLoader, which also requires
that we specify a batch_size and specify a Boolean value shuffle indicating whether or not we want the
DataLoader to shuffle the data on each epoch (pass through the dataset).
"""

from torch.utils.data import TensorDataset, DataLoader


def load_array(data_arrays, batch_size, is_train=True):
    dataset = TensorDataset(*(features, labels))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return dataloader


batch_size = 10
data_iter = load_array((features, labels), batch_size)

"""Now we can use data_iter in much the same way as we called the data_iter function in the previous
section. To verify that it’s working, we can read and print the first mini-batch of instances.
"""

for X, y in data_iter:
    print(X)
    print(y)
    break

"""##  Define the Model

![alt text](https://drive.google.com/uc?id=1-8TFMQ8pA1p4fpRTILCFXv_3X5a5RkST)
"""


class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.layer1 = torch.nn.Linear(2, 1, bias=True)

    def forward(self, x):
        y_pred = self.layer1(x)
        return y_pred


net = LinearRegressionModel()

"""##  Initialize Model Parameters
Before using net, we need to initialize the model parameters, such as the weights and biases in the linear
regression model. we specify that each weight parameter should be randomly sampled from a normal distribution with mean 0 and standard deviation 0.01.
The bias parameter will be initialized to zero by default. Both weight and bias will be attached with
gradients.
"""

net.layer1.weight.data = torch.Tensor(np.random.normal(size=(1, 2), scale=0.01, loc=0))
net.layer1.bias.data = torch.Tensor([0])

"""The code above looks straightforward but in reality something quite strange is happening here. We are
initializing parameters for a network even though we haven’t yet told nn how many dimensions the input
will have. It might be 2 as in our example or it might be 2,000, so we couldn’t just preallocate enough space
to make it work.
nn let’s us get away with this because behind the scenes, the initialization is deferred until the first time
that we attempt to pass data through our network. Just be careful to remember that since the parameters
have not been initialized yet we cannot yet manipulate them in any way.

##  Define the Loss Function
"""

loss = torch.nn.MSELoss(reduction="sum")

"""## Define the Optimization Algorithm
"""

trainer = torch.optim.SGD(net.parameters(), lr=0.03)

"""##  Training

You might have noticed that expressing our model through torch requires comparatively few lines of code.
We didn’t have to individually allocate parameters, define our loss function, or implement stochastic gradient descent. Once we start working with much more complex models, the benefits of relying on torch
abstractions will grow considerably. But once we have all the basic pieces in place, the training loop itself strikingly similar to what we did when implementing everything from scratch.
To refresh your memory: for some number of epochs, we’ll make a complete pass over the dataset
(train_data), grabbing one mini-batch of inputs and corresponding ground-truth labels at a time. 

For
each batch, we’ll go through the following ritual:

• Generate predictions by calling net(X) and calculate the loss l (the forward pass).

• Calculate gradients by calling l.backward() (the backward pass).

• Update the model parameters by invoking our SGD optimizer (note that trainer already knows which parameters to optimize over, so we just need to pass in the batch size.

For good measure, we compute the loss after each epoch and print it to monitor progress.
"""

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l_epoch = loss(net(features), labels)
    print('epoch {}, loss {}'.format(epoch + 1, l_epoch))

"""The model parameters we have learned and the actual model parameters are compared as below. We get
the layer we need from the net and access its weight (weight) and bias (bias). The parameters we have
learned and the actual parameters are very close.
"""

w = list(net.parameters())[0][0]
print('Error in estimating w', true_w.reshape(w.shape) - w)
b = list(net.parameters())[1][0]
print('Error in estimating b', true_b - b)

"""**Exercise**: You probably have asked yourself what zero_grad() is good for. It makes sure that the gradient is reset to zero after each gradient descent iteration. Without it, you would end up summing the gradients. As an exercise, wrap above code into a function and transform it to use batch gradient descent instead of mini-batch gradient descent. Return the loss after a single epoch, after updating the parameters. (Hint: You do not need a data_iter. You need to call the loss function two times)"""


def batch_gradient_descent(features, labels):
    ## write your code here
    num_epochs = 3
    for epoch in range(num_epochs):
        l = loss(net(features), labels) / len(features)
        trainer.zero_grad()
        l.backward()
        trainer.step()
        l_epoch = loss(net(features), labels)
        print('epoch {}, loss {}'.format(epoch + 1, l_epoch))
        # end of function


batch_gradient_descent(features, labels)
