#!/usr/bin/env python
# coding: utf-8

# # MCQs

# In[1]:


from MCQs import *


# # Coding

# In[2]:


from IPython.display import display, Markdown
from IPython.core.magic import register_cell_magic
from IPython.core.getipython import get_ipython
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


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


def within_array(x, y):
    return np.allclose(x, y)


# ## Q1

# Magic

# In[6]:


@register_cell_magic
def L00Q1(line, cell):

    correct = 'Hello World'

    globals = dict()
    exec(cell, globals)

    # Now we can check if something was done
    grt = globals.get('greet', None)

    if grt is None:
        print('You did not define a "greet" variable.')
    if (grt==correct):
        print('Correct')
    else:
        print('Incorrect')


# Question Cell

# In[7]:


def Code1():

    display(Markdown("""Using the code template given below, define a variable \'greet\'
    and store the string \'Hello  World\' in it. Print the variable."""))

    c = """%%L00Q1

# define the variable


#print the variable"""

    create_new_cell(c)


# In[25]:


# Code1()


# In[41]:


print('Code1() imported')


# ## Q2

# Magic

# In[9]:


@register_cell_magic
def L00Q2(line, cell):

    # correct answer
    def correct():
        return (np.sin(np.linspace(-1, 1, 20)))**2

    globals = dict()
    exec(cell, globals)

    # Now we can check if something was done
    Y = globals.get('y', None)
    X = globals.get('x', None)

    if X is None:
        print('Looks like you have changed the "x" variable. Use the original template variables.')
        return

    if(len(X)!=20):
        print('Incorrect. Note that you need to have 20 equidistant points.')
        return

    if Y is None:
        print('Looks like you have changed the "y" variable. Use the original template variables.')
        return

    if within_array(Y, correct()):
        print('Correct')
    else:
        print('Incorrect')


# Question Cell

# In[10]:


def Code2():

    display(Markdown("""Using the code template given below,
    Write a program that calculates the value of $sin^2(x)$ for 20 equidistant points between x = -1 and x = 1.
    Print the result. Note: Do not modify the template variables."""))

    c = """%%L00Q2

# import the required packages

# define the points
x =

# calculate sin^2(x)
y =

#print the variable y"""


    create_new_cell(c)


# In[30]:


# Code2()


# In[42]:


print('Code2() imported')


# ## Q3

# Magic

# In[37]:


@register_cell_magic
def L00Q3(line, cell):

    # correct answer
    def correct():

        x = np.linspace(-2, 2, 300)
        y = np.sin(x) * np.sin(x) - np.cos(x) * np.cos(x)

        plt.plot(x, y, 'k-', label = 'Correct', alpha = 0.5)
        plt.legend()
        return

    globals = dict()
    exec(strip_magic(line, cell), globals)

    X = globals.get('x', None)

    if X is None:
        print('Looks like you have changed the "x" variable. Use the original template variables.')
        return

    if(len(X)!=300):
        print('Incorrect. Note that you need to have 300 equidistant points.')
        return

    correct()


# Question Cell

# In[13]:


def Code3():

    display(Markdown("""Using the code template given below,
    Make a plot of $sin^2(x) - cos^2(x)$ for x going from -2 to 2,
    and having 300 points. Label the axes and the title."""))

    c = """%%L00Q3

# import the required packages


# define the points
x =

# calculate sin^2(x) - cos^2(x)
y =

#plot x vs y"""


    create_new_cell(c)


# In[40]:


# Code3()


# In[43]:


print('Code3() imported')
