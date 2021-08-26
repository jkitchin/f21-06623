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
from scipy.integrate import simps
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


# ## Q1

# In[5]:


def within_num(x, y):
    tol = 1e-6
    return not ((x < (y - tol)) or (y < (x - tol)))


# Magic

# In[6]:


@register_cell_magic
def L02Q1(line, cell):

    # correct answer
    def correct():

        points = 0

        for i in range(2, 10000):

            x = np.linspace(10, 20, i)
            y = x**3

            # analytical
            ya = x[-1]**4 / 4 - x[0]**4 / 4

            # np.trapz
            yt = np.trapz(y, x)

            # scipy.integrate.simps
            ys = simps(y, x)

            if(round(ya, 3)==round(yt, 3)==round(ys, 3)):
                points = i
                break

        return points

    globals = dict()
    exec(cell, globals)

    # Now we can check if something was done
    p = globals.get('points', None)

    if p is None:
        print('Looks like you have changed the "points" variable. Use the original template variables.')
        return

    elif within_num(p, correct()):
        print('Correct')
    else:
        print('Incorrect')


# Question

# In[7]:


def Code1():

    display(Markdown("""Integrate the function $f(x) = x^3$ analytically, using numpy.trapz and scipy.integrate.simps
    and compare the results, for limits of integration = [10, 20].
    Find the discretization required (number of points) such that all the three solutions are the same
    upto 3 decimal points. (Start with minimum 2 points)"""))

    c = """%%L02Q1
# import the required packages


# number of points required
points = 0

# iterate the solutions with an increasing number of points
for i in range(2, ):

    x =
    y = x**3

    # analytical solution
    y_a =

    # np.trapz
    y_t =

    # scipy.integrate.simps
    y_s =

    # check if the solution satisfies the question's criteria
    if():
        points = i
        break

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
def L02Q2(line, cell):

    # correct answer
    def correct():
        # analytical
        ya = 1 / 5 * ((1)**5 - (-1)**5)

        # gaussian quadrature
        def f(a):
            b = a**4
            return b

        w1 = (18 + np.sqrt(30)) / 36
        w2 = (18 - np.sqrt(30)) / 36
        x1 = np.sqrt(3/7 - 2/7*np.sqrt(6/5))
        x2 = -np.sqrt(3/7 - 2/7*np.sqrt(6/5))
        x3 = np.sqrt(3/7 + 2/7*np.sqrt(6/5))
        x4 = -np.sqrt(3/7 + 2/7*np.sqrt(6/5))

        yq = w1*(f(x1) + f(x2)) + w2*(f(x3)+f(x4))

        return yq

    globals = dict()
    exec(cell, globals)

    # Now we can check if something was done
    Yq = globals.get('y_q', None)

    if Yq is None:
        print('Looks like you have changed the "y_q" variable. Use the original template variables.')
        return

    elif within_num(Yq, correct()):
        print('Correct')
    else:
        print('Incorrect')


# Question

# In[11]:


def Code2():

    display(Markdown("""Integrate the function $f(x) = x^4$ analytically and using Gaussian Quadrature
    (https://en.wikipedia.org/wiki/Gaussian_quadrature) and compare the results, for limits of integration = [-1, 1]."""))

    c = """%%L02Q2
# import the required packages


# analytical
ya =

# gaussian quadrature
def f(a):

    return

w1 =
w2 =
x1 =
x2 =
x3 =
x4 =

y_q =

"""

    create_new_cell(c)


# In[12]:


# Code2()


# In[13]:


print('Code2() imported')
