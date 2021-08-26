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
from autograd import jacobian, grad
from scipy.integrate import quad


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
def L20Q1(line, cell):

    # correct answer
    def correct():

        def F(X):
            x, y = X
            return x**y, -y**x

        def r(t):
            return np.array([5*np.sin(t), 2*np.cos(t)])

        drdt = jacobian(r)

        def integrand(t):
            return F(r(t)) @ drdt(t)

        I, e = quad(integrand, 0.0, np.pi / 2)

        return I

    globals = dict()
    exec(strip_magic(line, cell), globals)

    # Now we can check if something was done
    Integral = globals.get('integral', None)

    if Integral is None:
        print('Looks like you have changed the "integral" variable. Use the original template variables.')
        return

    if within(Integral, correct()):
        print('Correct')
    else:
        print('Incorrect')


# Question

# In[7]:


def Code1():

    display(Markdown("""Evaluate the integral F(r) = [$x^y, -y^x$] , on the curve r(t) = [$5sin(t), 2cos(t)$]
    from t = 0 to $\pi/2$."""))

    c = """%%L20Q1
# import the required packages


# define the integral to be evaluated
def F():
    return

# define the curve
def r():
    return

drdt =

# function to be used with quad
def integrand():
    return

# quad function call
integral, e = quad()

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
def L20Q2(line, cell):

    # correct answer
    def correct():
        def f(x, y):
            return 5*x**2 + 10*x*y + 2*x*y**5 - 9

        dfdx = grad(f, 0)
        dfdy = grad(f, 1)

        def dydx(x, y):
            return -dfdx(x, y) / dfdy(x, y)

        ans = dydx(1.0, 2.0)

        return ans

    globals = dict()
    exec(strip_magic(line, cell), globals)

    # Now we can check if something was done
    Ans = globals.get('ans', None)

    if Ans is None:
        print('Looks like you have changed the "ans" variable. Use the original template variables.')
        return
    print(correct())
    if within(Ans, correct()):
        print('Correct')
    else:
        print('Incorrect')


# Question

# In[11]:


def Code2():

    display(Markdown("""Compute the derivative dy/dx for the function: 5x^2 + 10xy + 2xy^5 - 9$.
    Find the value of dy/dx at (1, 2)."""))

    c = """%%L20Q2
# import the required packages


# compute the derivative


# value at (1, 2)
ans =
"""

    create_new_cell(c)


# In[12]:


# Code2()


# In[13]:


print('Code2() imported')
