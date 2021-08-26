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

from scipy.integrate import solve_bvp
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# ## Supporting Questions

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


# # Q1

# In[5]:


def within_array(x, y):
    return np.allclose(x, y)


# Magic

# In[6]:


@register_cell_magic
def L09Q1(line, cell):

    # correct answer
    def correct():

        def bvp(x, U):
            u1, u2 = U
            du1dx = u2
            du2dx = -9 * u1
            return [du1dx, du2dx]

        def bc(Ua, Ub):
            u1a, u2a = Ua
            u1b, u2b = Ub
            return [u1a+11, u1b+11]

        X = np.linspace(1.5, 3.5)

        U1 = 0.5*X**2 - 11

        U2 = np.gradient(U1, X, edge_order=2)

        U = np.array([U1, U2])

        sol = solve_bvp(bvp, bc, X, U)
#         print(sol.message)
#         plt.plot(sol.x, sol.y[0])
#         plt.xlabel('x')
#         plt.ylabel('U')

        return sol.y[0]

    globals = dict()
    exec(cell, globals)

    BVP = globals.get('bvp', None)
    BC = globals.get('bc', None)

    if BVP is None:
        print('Looks like you have changed the "bvp" variable. Use the original template variables.')
        return
    if BC is None:
        print('Looks like you have changed the "bc" variable. Use the original template variables.')
        return

    X = np.linspace(1.5, 3.5)
    U1 = 0.5*X**2 - 11
    U2 = np.gradient(U1, X, edge_order=2)
    U = np.array([U1, U2])
    sol = solve_bvp(BVP, BC, X, U)

    if within_array(sol.y[0], correct()):
        print('Correct')
    else:
        print('Incorrect')


# Question

# In[7]:


def Code1():

    display(Markdown("""Solve the equation: $y'' + 9y = 0$, using solve_bvp, given that y(1.5) = y(3.5) = -11.
    In this question, only define the bvp() and bc() functions for the boundary value problem
    and the boundary conditions respectively."""))

    c = """%%L09Q1

def bvp():

    return []

def bc():

    return []
"""


    create_new_cell(c)


# In[8]:


# Code1()


# In[9]:


print('Code1() imported')


# # Q2

# Magic

# In[10]:


@register_cell_magic
def L09Q2(line, cell):

    # correct answer
    def correct():

        def bvp(x, U):
            u1, u2 = U
            du1dx = u2
            du2dx = -9 * u1
            return [du1dx, du2dx]

        def bc(Ua, Ub):
            u1a, u2a = Ua
            u1b, u2b = Ub
            return [u1a+11, u1b+11]

        X = np.linspace(1.5, 3.5)

        U1 = 0.5*X**2 - 11

        U2 = np.gradient(U1, X, edge_order=2)

        U = np.array([U1, U2])

        sol = solve_bvp(bvp, bc, X, U)
        plt.plot(sol.x, sol.y[0], 'k.', label = 'Correct', alpha = 0.3)
        plt.legend()

        return sol.y[0]

    globals = dict()
    exec(strip_magic(line, cell), globals)

    correct()


# Question

# In[11]:


def Code2():

    display(Markdown("""Solve the equation: $y'' + 9y = 0$, using solve_bvp, given that y(1.5) = y(3.5) = -11.
    In this question, focus on the initial guess and the solve_bvp() function call to plot the final answer (y vs x).
    You can either re-define bvp() and bc() or copy from Q1."""))

    c = """%%L09Q2
# import required packages


# define the boundary value problem
def bvp():

    return []

# boundary conditions
def bc():

    return []

# initial guess


# complete the solve_bvp() function call
sol = solve_bvp()

# plot
"""


    create_new_cell(c)


# In[12]:


# Code2()


# In[13]:


print('Code2() imported')
