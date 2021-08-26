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

import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize


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


# ## Q1

# Magic

# In[6]:


@register_cell_magic
def L17Q1(line, cell):

    # correct answer
    def correct():

        x1 = np.linspace(1, 10)
        x2 = np.sin(x1)
        y = x2**2 + x1
        plt.plot(x1, y)
        plt.plot(x1, y, 'k-', label = 'Correct function')

        Y = interp1d(x1, y, kind = 'quadratic', fill_value = 'extrapolate')
        xfit = np.linspace(-1, 11)
        plt.plot(xfit, Y(xfit), 'r--', label = 'Correct-Extrapolate')
        plt.legend()

        return

    globals = dict()
    exec(strip_magic(line, cell), globals)

    correct()



# Question

# In[7]:


def Code1():

    display(Markdown("""Use interp1d to find the function that fits to the data given below and plot the function for the range (-1, 11)."""))

    c = """%%L17Q1
# import the required packages



x = array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
y = array([ 1.70807342,  2.82682181,  3.01991486,  4.57275002,  5.91953576,
         6.07807302,  7.43163139,  8.97882974,  9.16984165, 10.29595897])


# interp1d call
fun = interp1d()


# plot


"""

    create_new_cell(c)


# In[8]:


# Code1()


# In[9]:


print('Code1() imported')


# ## Q2

# Magic

# In[10]:


@register_cell_magic
def L17Q2(line, cell):

    # correct answer
    def correct():

        def obj(X):
            x1, x2 = X
            y = x1**2 + 0.1*x2**3 - 5*np.sin(x1)
            return y

        guess = [1.0, 1.0]
        sol = minimize(obj, guess)

        hi = np.linalg.inv(sol.hess_inv)
        t = np.trace(hi)
        d = np.linalg.det(hi)

        return np.array([t, d])

    globals = dict()
    exec(strip_magic(line, cell), globals)

    # Now we can check if something was done
    Hess = globals.get('hess', None)
    Trace = globals.get('trace', None)
    Det = globals.get('det', None)

    if Trace is None:
        print('Looks like you have changed the "trace" variable. Use the original template variables.')
        return
    if Det is None:
        print('Looks like you have changed the "det" variable. Use the original template variables.')
        return

    Ans = np.array([Trace, Det])
    if within(Ans, correct()):
        print('Correct')
    else:
        print('Incorrect')


# Question

# In[11]:


def Code2():

    display(Markdown("""Find the hessian of the function $x_1^2 + 0.1x_2^3 - 5sin(x_1)$ at its local minimum value, which occurs at a positive value of x.
    Find the trace and determinant of hessian matrix at the stationary point. Note: Do not change the initial guess."""))

    c = """%%L17Q2
# import the required packages


# objective function
def obj():

    return

guess = [1.0, 1.0]

# minimize function call
sol = minimize()

# hessian matrix
hess =

# trace of the hessian matrix
trace =

# determinant of the hessian matrix
det =

"""

    create_new_cell(c)


# In[12]:


# Code2()


# In[13]:


print('Code2() imported')
