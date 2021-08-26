# -*- coding: utf-8 -*-
"""03_fode_1_coding.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1F1yEnYQV93eOYAuo-C2DKlsI942tQ9rh

# MCQs
"""
from .MCQs import *

"""# Coding"""

# Commented out IPython magic to ensure Python compatibility.
from IPython.display import display, Markdown
from IPython.core.magic import register_cell_magic
from IPython.core.getipython import get_ipython

import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import solve_ivp

"""## Supporting Functions"""

def strip_magic(line, cell):
    lines = cell.split('\n')
    stripped_lines = [line for line in lines if not line.strip().startswith('%')]

    if(len(lines)>len(stripped_lines)):
        print('Warning: The % magic does not work in this cell.')

    return ('\n'.join(stripped_lines))

def create_new_cell(contents):

    shell = get_ipython()
    shell.set_next_input(contents, replace=False)

"""## Q1

Magic
"""

@register_cell_magic
def L03Q1(line, cell):

    # correct answer
    def correct():

        x1 = 0
        x2 = 10
        X = np.linspace(x1, x2)
        y = np.zeros(X.shape)

        def f(x):
            return 2*(x-5)

        for i, val in enumerate(X):
            I,_ = quad(f, x1, val)
            y[i] = I

        plt.plot(X, y, 'k-.',label = 'Correct', alpha = 0.3)
        plt.legend()

        return

    globals = dict()
    exec(strip_magic(line, cell), globals)

    correct()

"""Question"""

def Code1():

    display(Markdown("""The differential equation for a parabola: $y = (x - 5)^2$ can be written as $\\frac{dy}{dx} = 2(x-5)$
    on taking its derivative. \n Make a plot of the integral values of $y' = 2(x-5)$ from x = 0 to x = 10,
    using scipy.integrate.quad. Is it the correct plot? Explain."""))

    c = """%%L03Q1
# import the required packages


x1 = 0
x2 = 10
X = np.linspace(x1, x2)
y = np.zeros(X.shape)

# complete the function
def f(x):

    return

# complete the code below
for i, val in enumerate(X):
    I,_ = quad()
    y[i] = I

# plot and label

"""


    create_new_cell(c)

# Code1()

print('Code1() imported')

"""## Q2

Magic
"""

@register_cell_magic
def L03Q2(line, cell):

    # correct answer
    def correct():

        def fs(x, y):
            dydx = 10 - np.exp(x)
            return dydx

        def event1(x, y):
            return y[0]

        event1.terminal = False

        y0 = np.array([-10])
        tspan = (-1, 5)

        sol = solve_ivp(fun = fs, t_span = tspan, y0 = y0, events = event1, t_eval = np.linspace(*tspan))

#         plt.plot(sol.t, sol.y.flatten())
        plt.plot(sol.t_events[0][0], sol.y_events[0][0], 'ko', label = f'$sol_1$: x = {round(sol.t_events[0][0], 1)}', alpha = 0.3)
        plt.plot(sol.t_events[0][1], sol.y_events[0][1], 'ko', label = f'$sol_2$: x = {round(sol.t_events[0][1], 1)}', alpha = 0.3)
        plt.legend()

        return

    globals = dict()
    exec(strip_magic(line, cell), globals)

    correct()

"""Question"""

def Code2():

    display(Markdown("""Make a plot of the integral values of $y' = 10 - e^x$ from x = -1 to x = 5,
    using scipy.integrate.solve_ivp. Using an event find the solutions to the equation;
    that is, when the integrated function crosses the x-axis. How many such events can you find. Plot all of them."""))

    c = """%%L03Q2

# import the required packages


# objective function
def fs():

    return

# event function
def event1():
    return

# complete the below line
event1.terminal =

y0 = np.array([-10])
tspan = (-1, 5)

# define solve_ivp() call
sol = solve_ivp( ,t_eval = np.linspace(*tspan))

# plot the solutions and label


"""


    create_new_cell(c)

# Code2()

print('Code2() imported')