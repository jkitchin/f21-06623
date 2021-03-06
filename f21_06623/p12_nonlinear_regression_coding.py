# -*- coding: utf-8 -*-
"""12_nonlinear_regression_coding.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sHkX8I0XxXxh1bsW_KwzEy4nrkfqqdJk

# MCQs
"""

from .MCQs import *

"""# Coding"""

# Commented out IPython magic to ensure Python compatibility.
from IPython.display import display, Markdown
from IPython.core.magic import register_cell_magic
from IPython.core.getipython import get_ipython

import numpy as np
from scipy.optimize import minimize
# %matplotlib inline
import matplotlib.pyplot as plt

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
def L12Q1(line, cell):

    # correct answer
    def correct():

        x = np.array([1.0, 1.45, 1.89, 2.34, 2.78, 3.23, 3.67, 4.12, 4.56, 5.0])
        y = np.array([.31, 0.5, 0.13, 0.061, 0.017, -0.0039, -0.009, -0.012, -0.011, -0.0066])

        # model equation
        def eqn(pars, x):
            yp = pars[0] + pars[1]*x + pars[2]*x**2
            return yp

        # objective function for sum squared errors
        def obj1(pars):
            pred = eqn(pars, x)
            err = y - pred
            sse = np.sum(err**2)
            return sse

        # objective function for absolute sum of errors
        def obj2(pars):
            pred = eqn(pars, x)
            err = y - pred
            abserr = np.sum(np.abs(err))
            return abserr

        # guess and minimize function calls
        guess = [0, 0, 0]
        sol1 = minimize(obj1, guess)
        sol2 = minimize(obj2, guess)

        # plot
        plt.plot(x, eqn(sol1.x, x), 'b.', label = 'Correct SSE')
        plt.plot(x, eqn(sol2.x, x), 'r.', label = 'Correct Abs_err')
        plt.legend()

        return

    globals = dict()
    exec(strip_magic(line, cell), globals)

    correct()

"""Question"""

def Code1():

    display(Markdown("""Using scipy.optimize.minimize, find the parameters for the equation $y = b_0 + b_1x + b_2x^2$, using the given data.

Compare the models obtained with 'sum squared errors' and 'sum of absolute errors' as objective functions.

Plot the final functions based on the parameters that you obtain."""))

    c = """%%L12Q1
# import the required packages


x = np.array([1.0, 1.45, 1.89, 2.34, 2.78, 3.23, 3.67, 4.12, 4.56, 5.0])
y = np.array([.31, 0.5, 0.13, 0.061, 0.017, -0.0039, -0.009, -0.012, -0.011, -0.0066])

# model equation
def eqn():

    return

# objective function for sum squared errors
def obj1():

    return

# objective function for absolute sum of errors
def obj2():

    return

# guess and minimize function calls
guess = [0, 0, 0]
sol1 =
sol2 =


# plot

"""

    create_new_cell(c)

# Code1()

print('Code1() imported')

"""## Q2

Magic
"""

@register_cell_magic
def L12Q2(line, cell):

    # correct answer
    def correct():

        x = np.array([1.0, 1.45, 1.89, 2.34, 2.78, 3.23, 3.67, 4.12, 4.56, 5.0])
        y = np.array([ 0.31 ,  0.076,  0.13,  0.061,  0.017, -0.0039, -0.009, -0.012, -0.011, -0.0066])
        w = np.array([2, 0.2, 2, 1, 2, 2, 2, 2, 2, 2])

        def obj(pars, x):
            b0, b1, b2 = pars
            Y = b0 + b1*x + b2*x**2
            return Y

        def loss(pars):
            err = (y - obj(pars, x))*w
            return np.sum(err**2)

        guess = [0.3, 0.13, 0]
        sol = minimize(loss, guess)

        Y = sol.x[0] + sol.x[1]*x + sol.x[2]*x**2
        plt.plot(x, Y, 'ko', label = 'Correct', alpha = 0.3)
        plt.legend()

        return

    globals = dict()
    exec(strip_magic(line, cell), globals)

    correct()

"""Question"""

def Code2():

    display(Markdown("""Using scipy.optimize.minimize, find the parameters for the equation $y = b_0 + b_1x + b_2x^2$, given the data:

x = [1.0, 1.45, 1.89, 2.34, 2.78, 3.23, 3.67, 4.12, 4.56, 5.0]

y = [0.31, 0.076, 0.13, 0.061, 0.017, -0.0039, -0.009, -0.012, -0.011, -0.0066]

Use sum squared error with weighted-regression for the weights given below:

w = [2, 0.2, 2, 1, 2, 2, 2, 2, 2, 2]

Plot the final function based on the parameters that you obtain."""))

    c = """%%L12Q2
# import the required packages


x = np.array([1.0, 1.45, 1.89, 2.34, 2.78, 3.23, 3.67, 4.12, 4.56, 5.0])
y = np.array([0.31, 0.076, 0.13, 0.061, 0.017, -0.0039, -0.009, -0.012, -0.011, -0.0066])

# complete the model
def eqn(pars, x):

    return

# complete the minimize objective function
def obj():

    return

# guess and minimize function call
guess =
sol = minimize()
print(sol.message)


# plot

"""

    create_new_cell(c)

# Code2()

print('Code2() imported')
