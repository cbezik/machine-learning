#Python code to implement k-means algorithm
#Author: Cody Bezik

import numpy as np

#Function to perform kmeans
def kmeans(data, k):
    1+1

#Loads a 2D dataset (e.g. toydata.txt)
def load2Ddata(filename):
    with open(filename) as f:
        lines = f.readlines()
    split_lines = list()
    for line in lines:
        split_lines.append(line.split(' '))
    cleaned_lines = list()
    for line in split_lines:
        cleaned_line = list()
        for element in line:
            element = element.replace("\n", "")
            cleaned_line.append(element)
        #print(cleaned_line)
        cleaned_line = filter(None, cleaned_line)
        #print(cleaned_line)
        cleaned_lines.append(cleaned_line)
    #print(cleaned_lines)
    x_data = [item[0] for item in cleaned_lines]
    y_data = [item[1] for item in cleaned_lines]
    #print(x_data, y_data)
    return(x_data, y_data)

#Here will be the calls to run k-means and produce output
x_points, y_points = load2Ddata("toydata.txt")
#print(x_points, y_points)
