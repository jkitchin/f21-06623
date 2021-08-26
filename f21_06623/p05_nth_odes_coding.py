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
from scipy.integrate import solve_ivp


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
#
# Reference: https://ocw.mit.edu/courses/mechanical-engineering/2-003-modeling-dynamics-and-control-i-spring-2005/readings/notesinstalment2.pdf

# Magic

# In[5]:


@register_cell_magic
def L05Q1(line, cell):

    # correct answer
    def correct():

        def dXdt(t, X):
            x, u = X
            dxdt = u
            dudt = -b/m *u - k/m *x
            return np.array([dxdt, dudt])

        k = 1
        m = 10
        b = 0.6

        tspan = (0, 500)
        y0 = np.array([1, 10])

        def max_x_event(t, X):
            x, u = X
            Xprime = dXdt(t, X)
            return Xprime[0]

        max_x_event.direction = -1   # event must go from positive to negative, i.e. a max

        sol = solve_ivp(dXdt, tspan, y0, t_eval = np.linspace(*tspan, 300), events=max_x_event, dense_output=True)

        te = sol.t_events[0]
        x, xp = sol.sol(te)

        plt.plot(te, x, 'ko', label = 'Correct Maxima', alpha = 0.3)
        plt.legend()

        return

    globals = dict()
    exec(strip_magic(line, cell), globals)

    correct()


# Question

# In[6]:


def Code1():

    display(Markdown("""The equation for a spring-mass system with a damper can be written as:
    $m\\frac{d^2x}{dt^2}+b\\frac{dx}{dt}+kx = 0$. \n\n The initial position of the mass: x = 1m and the initial velocity: x' = 10 m/s.
    The spring constant: k = 1 Kg.m/s^2, damping constant: b = 0.6 Kg/s, and the mass of the block: m = 10kg. \n\n Use solve_ivp() to identify and plot the maxima of the position x using an event function
    (atleast till amplitude drops below 1e-3)."""))

    c = """%%L05Q1
# import the required packages


# complete the objective function
 def dXdt():

    return np.array([])

k = 1
m = 10
b = 0.6

tspan = (0, 500)
y0 = np.array([1, 10])

# complete the event function
def max_x_event():

    return

max_x_event.direction =    # event must go from positive to negative, i.e. a max

# complete the solve_ivp() function call
sol = solve_ivp( , t_eval = np.linspace(*tspan, 300), )

# determine the solution for the questions
te = sol.t_events[0]
x, xp = sol.sol(te)

# plot



"""


    create_new_cell(c)


# In[7]:


# Code1()


# In[8]:


print('Code1() imported')


# ## Q2

# Magic

# In[9]:


@register_cell_magic
def L05Q2(line, cell):

    # correct answer
    def correct():

        for i in [1, 2, 3, 5, 10]:
            def dXdt(t, X):
                x, u = X
                dxdt = u
                dudt = -b/m *u - k/m *x
                return np.array([dxdt, dudt])

            k = i
            m = 10
            b = 0

            tspan = (0, 20)
            y0 = np.array([1, 10])
            sol = solve_ivp(dXdt, tspan, y0, t_eval = np.linspace(*tspan, 50))

            plt.plot(sol.y[0], sol.y[1], 'k.', label = f'k = {i}', alpha = 0.3)
            plt.axis('equal')

        return

    globals = dict()
    exec(strip_magic(line, cell), globals)

    correct()


# Question

# In[10]:


def Code2():

    display(Markdown("""Following up on Q1, make a plot of x vs x', for values of the spring constant
    k = [1, 2, 3, 5, 10] $Kg.m/s^2$, for tspan = (0, 20). Note: Do not worry about reaching uniform oscillation amplitude."""))

    c = """%%L05Q2
# import the required packages


for i in :
    # complete the objective function
    def dXdt(t, X):

        return np.array([])

    k = i
    m = 10
    b = 0

    tspan = (0, 20)
    y0 = np.array([1, 10])

    # complete the solve_ivp() function call
    sol = solve_ivp(, t_eval = np.linspace(*tspan, 500))

    # plot each case with labels


# add axes and plot titles


"""


    create_new_cell(c)


# In[11]:


# Code2()


# In[12]:


print('Code2() imported')
