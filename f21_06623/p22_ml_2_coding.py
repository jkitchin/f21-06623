#!/usr/bin/env python
# coding: utf-8

# # MCQs

# In[1]:


from .MCQs import *

# # Coding

# In[2]:


from IPython.display import display, Markdown
from IPython.core.magic import register_cell_magic
from IPython.core.getipython import get_ipython

import autograd.numpy as np
from scipy.optimize import least_squares
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import autograd.numpy.random as npr
from autograd.misc.optimizers import adam
from autograd import grad


# ## Supporting Functions

# In[3]:


def strip_magic(line, cell):
    lines = cell.split('\n')
    stripped_lines = [line for line in lines if not line.strip().startswith('%')]

    if(len(lines)>len(stripped_lines)):
        print('Warning: The % magic does not work in this cell.')

    return ('\n'.join(stripped_lines))


# In[4]:


def create_new_cell(contents):

    shell = get_ipython()
    shell.set_next_input(contents, replace=False)


# In[5]:


def within(x, y):
    return np.allclose(x, y)


# Supporting functions for the neural network.

# In[6]:


def nn(params, inputs, activation=np.tanh):
    """a neural network.
    params is a list of (weights, bias) for each layer.
    inputs goes into the nn. Each row corresponds to one output label.
    activation is the nonlinear activation function.
    """
    for W, b in params[:-1]:
        outputs = np.dot(inputs, W) + b
        inputs = activation(outputs)
    # no activation on the last layer
    W, b = params[-1]
    return np.dot(inputs, W) + b


# In[7]:


def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples, one for each layer."""
    return [(rs.randn(insize, outsize) * scale,   # weight matrix
             rs.randn(outsize) * scale)           # bias vector
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]


# In[8]:


def objective(params, step=None):
    pred = nn(params, np.array([X]).T)
    err = np.array([Y]).T - pred
    return np.mean(err**2)


# ## Q1

# Magic

# In[9]:


@register_cell_magic
def L22Q1(line, cell):

    # correct answer
    def correct():
        print('Running the solution code')
        # Data
        np.random.seed(3)
        X = np.linspace(-3, 3)
        Y = np.sin(X) *np.random.normal(1.0, 0.1, len(X))

        def nn2(params, inputs, activation):
            """a neural network.
            params is a list of (weights, bias) for each layer.
            inputs goes into the nn. Each row corresponds to one output label.
            activation is the nonlinear activation function.
            """
            for W, b in params[:-1]:
                outputs = np.dot(inputs, W) + b
                inputs = activation(outputs)
            # no activation on the last layer
            W, b = params[-1]
            return np.dot(inputs, W) + b

        def objective2(params, step=None):
            pred = nn2(params, np.array([X]).T, activation = cubic)
            err = np.array([Y]).T - pred
            return np.mean(err**2)

        def cubic(x):
            return x**3

        params2 = init_random_params(0.01, layer_sizes=[1, 3, 1])

        N = 50
        MAX_EPOCHS = 500

        for i in range(MAX_EPOCHS):
            params2 = adam(grad(objective2), params2,
                          step_size=0.01, num_iters=N)
            if objective2(params2, None) < 2e-5:
                break

        X2 = np.linspace(-3., 3)
        Z2 = nn2(params2, X2.reshape([-1, 1]), activation=cubic)
        plt.plot(X2, Z2, 'r.', label='Correct')
        plt.legend()

        return

    globals = dict()
    exec(strip_magic(line, cell), globals)

    correct()


# Question

# In[10]:


def Code1():

    display(Markdown("""Fit a neural network model to the data given in the below template, using a cubic activation function.
Plot the model's fit.

Given: maximum epochs = 500; num_iter = 50; neural network architecture: single layer with 3 neurons; learning rate: 0.01."""))

    c = """%%L22Q1
# import the required packages



# Data
np.random.seed(3)
X = np.linspace(-3, 3)
Y = np.sin(X) *np.random.normal(1.0, 0.1, len(X))

# neural network
def nn():

    return

# intialize parameters
def init_random_params():

    return

# objective function to check the fit and tolerance
def objective():

    return

# define activation function
def cubic():

    return

# NN hyperparameters and run the model
N = 50
MAX_EPOCHS = 500



# plot the model


"""

    create_new_cell(c)


# In[11]:


# Code1()


# In[12]:


print('Code1() imported')


# ## Q2

# Magic

# In[13]:


@register_cell_magic
def L22Q2(line, cell):

    # correct answer
    def correct():
        print('Running the solution code')
        # Data
        np.random.seed(8)
        X = np.linspace(-2, 2, 200)
        Y = np.sin(X) *np.random.normal(1.0, 0.2, len(X))

        # Splitting into train-test
        ind = np.arange(len(X))
        pind = np.random.permutation(ind)

        trainsplit = 0.2

        split = int(trainsplit * len(pind))

        train_ind = pind[:split]
        test_ind = pind[split:]

        train_x = X[train_ind]
        train_y = Y[train_ind]

        test_x = X[test_ind]
        test_y = Y[test_ind]

        # NN
        def nn(params, inputs, activation=np.tanh):
            """a neural network.
            params is a list of (weights, bias) for each layer.
            inputs goes into the nn. Each row corresponds to one output label.
            activation is the nonlinear activation function.
            """
            for W, b in params[:-1]:
                outputs = np.dot(inputs, W) + b
                inputs = activation(outputs)
            # no activation on the last layer
            W, b = params[-1]
            return np.dot(inputs, W) + b

        def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
            """Build a list of (weights, biases) tuples, one for each layer."""
            return [(rs.randn(insize, outsize) * scale,   # weight matrix
                     rs.randn(outsize) * scale)           # bias vector
                    for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]

        def objective(params, step=None):
            pred = nn(params, np.array([train_x]).T)
            err = np.array([train_y]).T - pred
            return np.mean(err**2)

        # training the NN
        params = init_random_params(0.01, layer_sizes=[1, 3, 1])

        N = 50
        MAX_EPOCHS = 200
        trainerror = []
        testerror = []

        for i in range(0, MAX_EPOCHS):
            params = adam(grad(objective), params,
                                  step_size=0.01, num_iters=N)
            if objective(params, None) < 2e-5:
                break

            msetrain = objective(params, None)
            msetest = np.mean((np.array([test_y]).T - nn(params, np.array([test_x]).T))**2)

            trainerror.append(msetrain)
            testerror.append(msetest)

        plt.plot(range(0, MAX_EPOCHS), trainerror, 'bx', label = 'Correct Train MSE', alpha = 0.3)
        plt.plot(range(0, MAX_EPOCHS), testerror, 'rx', label = 'Correct Test MSE', alpha = 0.3)
        plt.legend()

        return

    globals = dict()
    exec(strip_magic(line, cell), globals)

    correct()


# Question

# In[14]:


def Code2():

    display(Markdown("""Plot the training mean sqaured error and test mean squared error vs the number of epochs
    for a neural network with the given data and hyperparameters.

Given: max_iter = 50, activation function = tanh, train-test split: 80-20, neural network architecture: single layer with 3 neurons; learning rate: 0.01."""))

    c = """%%L22Q2
# import the required packages



# Data
np.random.seed(8)
X = np.linspace(-2, 2, 200)
Y = np.sin(X) *np.random.normal(1.0, 0.2, len(X))

# split the data



# NN functions
def nn():

    return

def init_random_params():

    return

def objective():

    return

# Training the NN
N = 50
MAX_EPOCHS = 200




# plot



"""

    create_new_cell(c)


# In[15]:


# Code2()


# In[16]:


print('Code2() imported')
