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
from autograd import elementwise_grad
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from autograd import jacobian


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
def L19Q1(line, cell):

    # correct answer
    def correct():

        def f(x):
            return np.sin(x)

        x = np.linspace(-1.5, 4.5)
        y = f(x)
        df = elementwise_grad(f)
        yp = df(x)

        plt.plot(x, yp, 'k.', label = 'Correct', alpha = 0.3)
        plt.legend()

        return

    globals = dict()
    exec(strip_magic(line, cell), globals)

    correct()


# Question

# In[7]:


def Code1():

    display(Markdown("""Plot the derivative of sin(x) between the points x = -1.5 to x = 4.5,
    using elementwise_grad."""))

    c = """%%L19Q1
# import the required packages


# define the function to take the derivative
def f():
    return

x = np.linspace(-1.5, 4.5)
y = f(x)

# use elementwise_grad to take the derivative


# plot the derivative

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
def L19Q2(line, cell):

    # correct answer
    def correct():

        # transformation 1
        def f1(P):
            u, v = P
            return np.array([np.sin(u), np.cos(v)])

        jf1 = jacobian(f1)

        def integrand1(v, u):
            J = jf1(np.array([u, v]))
            return np.linalg.det(J) * (np.sin(u)**3 + np.sin(u)*np.cos(v) + np.cos(v)**5)

        # integrand(y, x)
        xa, xb = -np.pi/2, np.pi/2
        ya, yb = np.pi/2, 0

        integral1, _ = dblquad(integrand1, xa, xb, ya, yb)

        # transformation 2
        def f2(P):
            u, v = P
            return np.array([np.tan(u), np.sin(v)])

        jf2 = jacobian(f2)

        def integrand2(v, u):
            J = jf2(np.array([u, v]))
            return np.linalg.det(J) * (np.tan(u)**3 + np.tan(u)*np.sin(v) + np.sin(v)**5)

        # integrand(y, x)
        xa, xb = -np.pi/4, np.pi/4
        ya, yb = 0, np.pi/2

        integral2, _ = dblquad(integrand2, xa, xb, ya, yb)
        return np.array([integral1, integral2])

    globals = dict()
    exec(strip_magic(line, cell), globals)

    Int1 = globals.get('integral1', None)
    Int2 = globals.get('integral2', None)

    if Int1 is None:
        print('Looks like you have changed the "integral1" variable. Use the original template variables.')
        return

    if Int2 is None:
        print('Looks like you have changed the "integral2" variable. Use the original template variables.')
        return

    sol = np.array([Int1, Int2])

    if within(sol, correct()):
        print('Correct')
    else:
        print('Incorrect')


# Question

# In[11]:


def Code2():

    display(Markdown("""Prove that the choice of transformation for a change of variables does not affect the value of an integral.
    Integrate the function: $x^3 + xy + y^5$, from x = -1 to 1 and y = 0 to 1. Use the following transformations:

Transformation 1: x = sin(u); y = cos(v)

Transformation 2: x = tan(u); y = sin(v)"""))

    c = """%%L19Q2
# import the required packages



# transformation 1

# jacobian


# integrand for dblquad
def integrand1():

    return

# limits of integration


# dblquad function call
integral1, _ = dblquad()


# transformation 2

# jacobian


# integrand for dblquad
def integrand2():

    return

# limits of integration


# dblquad function call
integral2, _ = dblquad()
"""

    create_new_cell(c)


# In[12]:


# Code2()


# In[13]:


print('Code2() imported')
