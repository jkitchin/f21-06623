#!/usr/bin/env python
# coding: utf-8

# # MCQs

# In[1]:


from urllib.request import urlopen
url = 'https://drive.google.com/uc?id=1vZZBZ9UOBhjMSAw_viB6m4x5R6I52zYa'
py = urlopen(url).read().decode('utf-8')
exec(py)


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


def within_num(x, y):
    tol = 1e-6
    return not ((x < (y - tol)) or (y < (x - tol)))


# ## Q1

# Magic

# In[6]:


@register_cell_magic
def L01Q1(line, cell):
    
    # correct answer 
    def correct(x):
        import numpy as np
        return np.sqrt(x)
    
    globals = dict()
    exec(cell, globals)
    
    # Now we can check if something was done
    sqrt = globals.get('square_root', None)
    n = globals.get('num', None)
    
    if n is None:
        print('You did not define a "num" variable.')
        return
    
    if sqrt is None:
        print('You did not define a "square_root" variable.')
    elif within_num(sqrt, correct(n)):
        print('Correct')
    else:
        print('Incorrect')


# Question

# In[7]:


def Code1():
    
    display(Markdown('Create a function that returns the square root of a number.'))
    
    c = """%%L01Q1
# import the required packages


def f(x):
    ans =     # complete the line
    return ans
    
num =     # enter any number to find the square root
square_root =      # call the function

print(f'Your answer: {square_root}')"""
    
    create_new_cell(c)


# In[8]:


# Code1()


# In[16]:


print('Code1() imported')


# ## Q2

# Magic

# In[9]:


@register_cell_magic
def L01Q2(line, cell):
    
    # correct answer 
    def correct(x, y):
        return x+y
    
    globals = dict()
    exec(cell, globals)
    
    # Now we can check if something was done
    s1 = globals.get('string1', None)
    s2 = globals.get('string2', None)
    s = globals.get('string_', None)
    
    if s1 is None:
        print('Looks like you have changed the "string1" variable. Use the original template variables.')
        return
    if s2 is None:
        print('Looks like you have changed the "string2" variable. Use the original template variables.')
        return
    if s is None:
        print('Looks like you have changed the "string_" variable. Use the original template variables.')
        return
    
    if correct(s1, s2)==s:
        print('Correct')
    else:
        print('Incorrect')


# Question

# In[10]:


def Code2():
    
    display(Markdown("""Create a program that takes 2 strings and returns a single string 
    that has string 1 as the first half and string 2 as the later."""))
    
    c = """%%L01Q2
# define the two strings
string1 = 
string2 = 

# combine the strings and assign to the variable 'string_'

"""
    
    create_new_cell(c)


# In[11]:


# Code2()


# In[17]:


print('Code2() imported')


# ## Q3

# In[12]:


def within_array(x, y):
    return np.allclose(x, y)


# Magic

# In[13]:


@register_cell_magic
def L01Q3(line, cell):
    
    # correct answer 
    def correct(n1, n2):
        q = n1//n2
        d = n1%n2
    
        return q, d
    
    globals = dict()
    exec(cell, globals)
    
    # Now we can check if something was done
    n1 = globals.get('num1', None)
    n2 = globals.get('num2', None)
    Q = globals.get('Q', None)
    R = globals.get('R', None)
    
    if n1 is None:
        print('Looks like you have changed the "num1" variable. Use the original template variables.')
        return
    if n2 is None:
        print('Looks like you have changed the "num2" variable. Use the original template variables.')
        return
    if Q is None:
        print('Looks like you have changed the "Q" variable. Use the original template variables.')
        return
    if R is None:
        print('Looks like you have changed the "R" variable. Use the original template variables.')
        return
    
    if within_array(np.array([Q, R]), np.array(correct(n1, n2))):
        print('Correct')
    else:
        print('Incorrect')


# Question

# In[14]:


def Code3():
    
    display(Markdown("""Create a program that takes 2 numbers and returns
    the quotient and dividend of number1 divided by number2."""))
    
    c = """%%L01Q3
# define the two numbers
num1 = 
num2 = 

# assign the quotient and remainder to the variables defined below
Q = 
R = 

"""
    
    create_new_cell(c)


# In[15]:


# Code3()


# In[18]:


print('Code3() imported')

