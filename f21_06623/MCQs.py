#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime as DT
from datetime import date, timedelta, datetime
import pandas as pd
import numpy as np
from IPython.core.getipython import get_ipython


# google spreadsheet with all lectures. Contains fields: date, tag, lecture, question, answer. Below link is a modified url of the google spreadsheet. Need to keep sharing permission to 'anyone with the link can open'.

# In[2]:


url = 'https://docs.google.com/spreadsheets/d/1qSaBe73Pd8L3jJyOL68klp6yRArW7Nce/export?format=xlsx&gid=1923176268'
df = pd.read_excel(url)


# Dictionary of lecture dates and names; Date is not currently used anywhere.

# In[3]:


lectures = {'08/31/2020':'00_intro',
           '09/02/2020': '01_jupyter',
           '09/09/2020': '02_integration_1',
           '09/14/2020': '03_fode_1', 
            '09/16/2020': '04_fode_2',
            '09/21/2020': '05_nth_odes',
            '09/23/2020': '07_nla_1',
            '09/28/2020': '08_nla_2',
            '09/30/2020': '09_bvp',
            '10/05/2020': '10_min_max',
            '10/07/2020': '11_regression',
            '10/12/2020': '12_nonlinear_regression',
            '10/14/2020': '13_constrained_optimization',
            '10/19/2020': '15_intro_linear_algebra',
            '10/21/2020': '16_linear_algebra',
            '10/26/2020': '17_linear_algebra_2',
            '10/28/2020': '18_linear_regression',
            '11/02/2020': '19_introduction_to_autograd',
            '11/04/2020': '20_autograd_applications',
            '11/09/2020': '21_machine_learning',
            '11/16/2020': '22_ml_2',
            '11/18/2020': '23_gp'}

# lecture names are used as tags
listOfLectures = list(lectures.values())


# ## Supporting Functions

# In[4]:


def create_new_cell(contents):     # does not work in Colab
    
    shell = get_ipython()
    shell.set_next_input(contents, replace=False)


# In[5]:


def find_tags(keys):
    
    # Empty dataframe to which tagged rows are added
    tdf = pd.DataFrame(columns = lec.columns)
    
    for k in keys:     # for multiple tags as input
        for i in range(0, len(lec)):
            if(k in lec.iloc[i].tag):
                tdf = tdf.append(lec.iloc[i])
    return tdf


# In[6]:


def reset():
    global old_q     # to record questions already answered for ensuring no repeats
    old_q = []   
    global count     # to count the number of questions which were correctly answered
    count = []  
    global current_tag   # to record a change in tag
    current_tag = ''
reset()


# In[7]:


def TAG_DF(keys=[]):     # to handle multiple tags as input 

    global current_tag, count, lec

    check = 1
    # display all tags
    if('tags' in keys):
        print(f'Find a question on:\n{list(set(tags))}')
        current_tag = keys
        check=0
        return None, check
    
    if(set(keys).issubset(set(tags))==False or keys==[]):     # if tag is invalid
        print('Tags Invalid; Type MCQ([\'tags\']) to see all the available tags')
        check = 0
        return None, check
    else:                          # if tag is valid   
        
        # handling a change of tag, if requested before completing all the questions from previous tag
        if(set(keys).issubset(set(current_tag))==False and len(count)!=0 and current_tag!=['tags'] or len(current_tag)!=len(keys) and len(count)!=0):
            print('There are unanswered questions for the last tag\s. Do you still want to change the tags? (y/n)')       
            change = input()
            if(change=='y'):
                reset()
            else:
                keys=current_tag   
        
        if('all' in set(keys)):
            if(len(lec)>0):
                tagged_df = lec
                current_tag = keys
            else: 
                print(f'Select a lecture from \n{listOfLectures} \n\nand run lecName([\'lecture_file_name\'])')
                check=0
                return None, check
            
        else:
            tagged_df = find_tags(keys)     # data frame with just the tagged questions
            current_tag = keys
            
    return tagged_df, check


# In[8]:


def nameTags(l):
    t = [x.split() for x in list(l.tag)]   # all the tags available in the selected dataframe
    tags = [x1 for x2 in t for x1 in x2]    # flattening the nested list t (t is nested due to multiple tags for each question)
    tags.append('all')
    return tags


# In[9]:


def lecName(keys=None):   
    
    """ Selects the lecture from which questions will be presented in this file.
    
    Parameters:
    ___________
    
    keys: (list of string/s): ['lecture_file_name/s'] or ['available'] or ['all']
    
    if ['lecture_file_name/s']: generates a dataframe only with the lectures included in the list 
                                eg. lecName(['00_intro'];
    if ['available']: prints all the available lecture names;
    if ['all']: generates a dataframe with all the lectures.
    """
    
    global lec
    
    if(keys==['available']):
        print(f'Select a lecture from \n{listOfLectures} \n\nand run lecName([\'lecture_file_name\']).')
        return
        
    if(keys==['all']):
        lec = df
        print(f'Imported MCQs for lecture {keys}.')
        return

    lec = df[df['lecture file'].isin(keys)]
    print(f'Imported MCQs for lecture {keys}.')
    
    return # lec

# default
lec = pd.DataFrame()


# ## Presenting Questions

# In[10]:


def MCQ(keys=[]):
    
    """Prints multiple choice questions based on the selected tags in the chosen lectures.
    
    Parameters:
    ___________
    
    keys: (list of string/s): ['tags'] or ['tag_name/s'] or ['all']
    
    if ['tags']: prints the available tags;
    if ['tag_name/s']: presents questions only based on the selected tags;
                       eg. MCQ(['numpy', 'scipy'])
    if ['all']: presents questions for all the tags available from the selected lectures;
    
    """

    global lec, tags, current_tag
    
    # checks that a lecture has been selected, otherwise asks to do so
    if(len(lec)<1):
        print(f'Select a lecture from \n{listOfLectures} \n\nand run lecName([\'lecture_file_name\'])')
        return
   
    # collects all the tags from the chosen lecture/s
    tags = nameTags(lec)
    
    # dataframe with just tagged questions
    filtered_df, check = TAG_DF(keys)
    if(check==0):
        return
   
    tagged_df = filtered_df
    
    # randomizing the questions
    rand = list(np.random.choice(len(tagged_df), size = len(tagged_df), replace = False))

    # selecting just the unanswered questions
    next_q = [x for x in rand if x not in old_q]
    
    # first question in the list p will be printed, saving it in old_q to avoid repeating in the next run
    old_q.append(next_q[0])

    # printing 
    n = tagged_df.iloc[next_q[0]]
    
    q = f'Q.({len(old_q)}/{len(tagged_df)})\n{n.question}\n'
    print(q)
    
    options = f'a) {n.a}\nb) {n.b}\nc) {n.c}\nd) {n.d}\n'
    print(options)
    
    # for answering multiple times until correct
    counter = 0
    
    feed = input()
    while(counter==0):
        if(feed == n.answer):
            print('Correct')
            count.append(1)
            counter = 1
            if(len(old_q)<len(tagged_df)):
                print('Run the cell again for the next question')
                create_new_cell(f'MCQ({keys})')
            else:
                print(f'No more questions for {current_tag}.')
                create_new_cell('MCQ([\'tags\'])')
                reset()
        else:
            print('Incorrect; Try again')
            feed = input()


# In[11]:


# print('MCQ() imported')

