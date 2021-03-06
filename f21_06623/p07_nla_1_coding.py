# -*- coding: utf-8 -*-
"""07_nla_1_coding.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1S6tGEDzEeBJyFzY0Dq86p7ekwDxIxVPh

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
from scipy.optimize import fsolve
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
def L07Q1(line, cell):

    # correct answer
    def correct():

        def obj(x, c):
            z = np.exp(x) - c*x
            return z

        const = np.linspace(10, 30)
        guess = 10

        ans = [fsolve(obj, guess, args = (c, ))[0] for c in const]

        plt.plot(const, ans, 'k.', alpha = 0.3, label = 'Correct')
        plt.legend()

        return

    globals = dict()
    exec(strip_magic(line, cell), globals)

    correct()

"""Question"""

def Code1():

    display(Markdown("""Solve the equation $e^x - cx$, for c between the range (10, 30).
    Use list comprehesion with fsolve to store the answers in a single list.
    Plot this list against the c-values."""))

    c = """%%L07Q1
# import the required packages


# complete the objective function
def obj(x, c):
    z = np.exp(x) - c*x
    return z

const = np.linspace(10, 30)
guess = 10

# complete the list comprehension line
ans = []


# plot



"""


    create_new_cell(c)

# Code1()

print('Code1() imported')

"""## Q2

Magic
"""

@register_cell_magic
def L07Q2(line, cell):

    # correct answer
    def correct():

        def f(x, Y):
            y1, y2 = Y
            dy1dx = y2
            dy2dx = 2
            return np.array([dy1dx, dy2dx])

        def obj(x):
            y0 = np.array([25, -10])
            tspan = (0, x)
            sol = solve_ivp(f, tspan, y0)
            return sol.y[0][-1] - Y

        Y = 10
        guess1 = 1
        ans1 = fsolve(obj, guess1)
        guess2 = 9
        ans2 = fsolve(obj, guess2)

        y0 = np.array([25, -10])
        sol = solve_ivp(f, (0, 10), y0, t_eval=np.linspace(0, 10))

#         plt.plot(sol.t, sol.y[0])
        plt.plot(ans1, Y, 'ko', label = f'x1 = {ans1}')
        plt.plot(ans2, Y, 'ko', label = f'x2 = {ans2}')
        plt.legend()


        return

    globals = dict()
    exec(strip_magic(line, cell), globals)

    correct()

"""Question"""

def Code2():

    display(Markdown("""Find the value of x when y = 10, for the equation $\frac{d^2y}{dx^2} = 2$,
    given that y(0) = 25 and y'(0) = -10. Use fsolve and solve_ivp (without event).
    How many solutions are there? Plot all the solutions on an x vs y plot."""))

    c = """%%L07Q2
# import the required packages


# complete the solve_ivp() objective function
def f(x, Y):

    return np.array([])

# complete the fsolve() objective function
def obj():
    y0 = np.array([25, -10])
    tspan = (0, x)
    sol = solve_ivp()
    return

Y = 10
guess1 =
ans1 = fsolve()
guess2 =
ans2 = fsolve()

y0 = np.array([25, -10])
sol = solve_ivp(f, (0, 10), y0, t_eval=np.linspace(0, 10))


# add the plots for the solutions
plt.plot(sol.t, sol.y[0])
plt.axhline(10, color='k', linestyle='--')
plt.legend()


"""


    create_new_cell(c)

# Code2()

print('Code2() imported')
