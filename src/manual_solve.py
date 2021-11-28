#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/python

"""
NUI Galway CT5132 Programming and Tools for AI

Assignment 3

Student name:Arunima Dy
Student ID: 21230125

My Github repo: https://github.com/arunimadey23/ARC

"""

import os, sys
import json
import numpy as np
import re

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.

def solve_67e8384a(x):
    '''
    
    On analying this, we can see that all 3*3 matrix is converted to 6*6 matrix 
    wherein the input matrix is flipped along its horizontal and vertical axis.
    
    Here, we first take the input array and then flip it across vertical axis 
    and append the output with the input array. Next we flip the resulting array 
    in horizontal axis and the final output array i appended to the previous array
    
    The training and test datasets give correct results for this algorithm.
    
    '''

    x_copy=x.copy() # create copy of input array so that original array remains unchanged
    f=np.flip(x_copy,axis=1) # flip the input array vertically
    s=np.append(x_copy,f,axis=1) # append the flipped array to existing array
    fud=np.flip(s) # flip the appended array vertically
    x_copy=np.append(s,fud,axis=0) # append the final flipped array

    return x_copy
   

def solve_496994bd(x):
    '''
    
    On analying this, we can see that the mirror image of the coloured cells have been
    superimposed at the end of the matrix. There are even number of rows in each input 
    array, so we can divide the input array into equal number of chunks.
    
    Here, we split the input array into two chunks. We take the first chunk which need 
    to be superimposed and flip it upside down. Thereafter, we append the flipped array 
    to the first chunk of the input array.
    
    The training and test datasets give correct results for this algorithm.
    
    '''
    
    x_copy=x.copy() # create copy of input array so that original array remains unchanged
    chunks=(np.array_split(x_copy,2)) # divide the input array into two equal chunks
    f=np.flipud(chunks[0]) # flip the first chunk upside down
    x_copy=np.append(chunks[0],f,axis=0) # append the flipped array to the first chunk
    
    return x_copy

def solve_47c1f68c(x):
    '''
    
    On analying this, we can see the input array is divided into four quadrants and
    the image appearing in the first quadrant is superimposed in each quadrant where it's
    horizontally and vertically flipped. The '+' which divides the array in four quadrants 
    is removed from resulting array and it's colour is filled in the resulting images.
    
    Here, we divide the input array into four chunks and take the first quadrant. The first
    chunk is refined into two arrays, first stores the marix of the image and second stores 
    the '+' symbol as we need the colour. The first chunk is flipped vertically and appended
    to the existing chunk and then the entire chunk is flipped vertically. We get the indices 
    of the resulting for the non zero values and those values are replaced with the colour of 
    '+'.
    
    The training and test datasets give correct results for this algorithm.
    
    '''    
    
    x_copy=x.copy() # create copy of input array so that original array remains unchanged
    chunks1=(np.array_split(x_copy,2)) # split array horizontally
    chunks2=np.array_split(chunks1[0],2,axis=1) # split array vertically
    c=chunks2[0] # take the first quadrant chunk
    c_sub=c[0:-1,0:-1] # refine the first quadrant chunk by removing the '+'
    c_extra=c[:,-1] # store the '+' for colour 
    f=np.flip(c_sub,axis=1) # flip the first quadrant
    s_app_col=np.append(c_sub,f,axis=1) # append the flipped quadrant to the original chunk
    fud=np.flipud(s_app_col) # flip the resulting array upside down
    s_app_row=np.append(s_app_col,fud,axis=0) # append to existing array
    x_copy=np.where(s_app_row!=0,c_extra[0],s_app_row) # find non zero indices and replace with colour of '+' 
    
    return x_copy


def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    
def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    if y.shape != yhat.shape:
        print(f"False. Incorrect shape: {y.shape} v {yhat.shape}")
    else:
        print(np.all(y == yhat))


if __name__ == "__main__": main()

