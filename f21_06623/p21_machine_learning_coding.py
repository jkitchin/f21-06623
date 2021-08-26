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

# In[10]:


@register_cell_magic
def L21Q1(line, cell):

    # correct answer
    def correct():
        print('Running the solution code')
#         # Data
#         np.random.seed(3)
#         X = np.linspace(0, 5*np.pi, 100)
#         Y = 3*np.sin(X) * np.random.normal(1.0, 0.5, len(X)) + X

#         N = 50
#         MAX_EPOCHS = 100

#         neurons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100, 200]
#         sumsquared = []

#         for n in neurons:
#             params = init_random_params(0.1, layer_sizes=[1, n, 1])
#             print('n', n)
#             for i in range(MAX_EPOCHS):
#                 params = adam(grad(objective), params,
#                               step_size=0.001, num_iters=N)
#                 if objective(params, _) < 2e-5:
#                     break

#             sumsquared.append(objective(params))

#         plt.plot(neurons, sumsquared, 'r.', label = 'Correct')

        solx = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100, 200])
        soly = [10.8619915176313, 5.8476713228725385, 4.541839262408973, 4.003736393154463, 4.241890104633619, 4.283135570960502,
                4.684531390186648, 4.275154641355541, 4.737285654599213, 4.73896267052014, 4.739039006669549,
                4.751135315823153, 4.788037370881726, 4.772755894180145, 4.112843093753394, 3.558614672944332]
        plt.plot(solx, soly, 'r.', label = 'Correct')
        plt.legend()

        return

    globals = dict()
    exec(strip_magic(line, cell), globals)

    correct()


# Question

# In[11]:


def Code1():

    display(Markdown("""Fit a single layer neural network model to the data given in the below template.
    Make a plot of the objective function value (mean squared error) vs number of neurons for the range:
    neurons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100, 200].

    Keep other parameters constant. This problem might take 4-5 minutes to run.

    Given: step_size=0.001, maximum epochs = 100; num_iter = 50; neural network architecture: single layer with n neurons; activation function: tanh"""))

    c = """%%L21Q1
# import the required packages



# Data
# np.random.seed(3)
# X = np.linspace(0, 5*np.pi, 100)
# Y = 3*np.sin(X) * np.random.normal(1.0, 0.5, len(X)) + X

# neural network
def nn():

    return

# intialize parameters
def init_random_params():

    return

# objective function to check the fit and tolerance
def objective():

    return



# NN hyperparameters and run the model
N = 50
MAX_EPOCHS = 100



# plot the mean squared errors vs number of neurons


"""

    create_new_cell(c)


# In[12]:


# Code1()


# In[13]:


print('Code1() imported')


# ## Q2

# Magic

# In[14]:


@register_cell_magic
def L21Q2(line, cell):

    # correct answer
    def correct():
        print('Running the solution code')
        # Data
        np.random.seed(3)
        X = np.linspace(0, np.pi, 20)
        Y = np.sin(X) * np.random.normal(1.0, 0.1, len(X))

        def objectiveq2(params, step=None):
            pred = nn(params, np.array([X]).T)
            err = np.array([Y]).T - pred
            return np.mean(err**2)

        N = 50
        MAX_EPOCHS = 200
        sumsquared = []
        params = init_random_params(0.1, layer_sizes=[1, 3, 1])

        for i in range(MAX_EPOCHS):
            params = adam(grad(objectiveq2), params,
                        step_size=0.001, num_iters=N)
            if objectiveq2(params, None) < 2e-5:
                break

            SSerrors = objectiveq2(params, None)
            sumsquared.append(SSerrors)

        plt.plot(range(0, MAX_EPOCHS), sumsquared, 'r--', label = 'Correct')
        plt.legend()
        return

    globals = dict()
    exec(strip_magic(line, cell), globals)

    correct()


# Question

# In[15]:


def Code2():

    display(Markdown("""Fit a neural network model to the data given in the below template.
    Make a plot of the mean squared errors vs the number of epochs, upto 200 epochs.
    Keep other parameters constant. This problem might take a minute to run.

Given: step_size = 0.001; num_iter = 50; neural network architecture: single layer with 3 neurons; activation function: tanh."""))

    c = """%%L21Q2
# import the required packages



# Data
np.random.seed(3)
X = np.linspace(0, np.pi, 20)
Y = np.sin(X) * np.random.normal(1.0, 0.1, len(X))

# neural network
def nn():

    return

# intialize parameters
def init_random_params():

    return

# objective function to check the fit and tolerance
def objective():

    return



# NN hyperparameters and run the model
N = 50
MAX_EPOCHS = 200



# plot the mean squared errors vs epochs




"""

    create_new_cell(c)


# In[16]:


# Code2()


# In[17]:


print('Code2() imported')
