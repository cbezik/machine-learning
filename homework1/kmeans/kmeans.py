#Python code to implement k-means algorithm
#Author: Cody Bezik

import numpy as np
import random

#Function to perform kmeans
def kmeans(data, k):
    #Data will be represented in the following way: a list of all points, each point being a list containing the value in each dimension and its final value being an integer identifying the cluster it is assigned to
    data_clustered = list()
    for line in data:
        line.append(0)
        data_clustered.append(line)
    #print(data_clustered)
    clusters = list()
    point_dimension = len(data_clustered[0]) - 1 #Requires all data points be same size
    #print(point_dimension)
    #Get maximum data point for scaling random numbers
    max_data_scale = np.amax(np.absolute(np.asarray(data_clustered)))
    #max_data_scale = max(max_data_list)
    #print(max_data_list)
    #print(max_data_scale)
    #Initialize random clusters
    cluster_id = 0
    rand_clusters = list()
    for x in range(0, k):
        #print("Cluster test")
        rand_centers = list()
        for y in range(0, point_dimension):
            rand = (2.0*random.random() - 1.0)*max_data_scale
            rand_centers.append(rand)
        #print(rand_centers)
        rand_centers.append(cluster_id)
        rand_clusters.append(rand_centers)
        cluster_id += 1
    #print(rand_clusters)


#Loads a dataset (e.g. toydata.txt)
def loaddata(filename):
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
    converted_lines = list()
    for line in cleaned_lines:
        converted_line = list()
        for element in line:
            element = float(element)
            converted_line.append(element)
        converted_lines.append(converted_line)
    #print(cleaned_lines)
    #x_data = [item[0] for item in cleaned_lines]
    #y_data = [item[1] for item in cleaned_lines]
    #print(x_data, y_data)
    #print(converted_lines)
    return(converted_lines)

#Here will be the calls to run k-means and produce output
data = loaddata("toydata.txt")
#print(data)
#Run k-means on the toy data for 3 clusters
kmeans(data, 3)
