#Python code to implement k-means algorithm
#Author: Cody Bezik

import numpy as np
import random
import math
import copy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

#Calculates distances between a point and a cluster
def point_to_cluster_distance(point, cluster):
    temp_point = copy.deepcopy(point)
    temp_cluster = copy.deepcopy(cluster)
    temp_point.pop()
    temp_cluster.pop()
    distance = np.linalg.norm(np.asarray(temp_point) - np.asarray(temp_cluster))
    """for i in range(0, len(temp_point)):
            in temp_point:
        for cluster_value in temp_cluster:
            distance += (point_value - cluster_value)**2
    distance = math.sqrt(distance)"""
    return distance

#Tolerance checker - checks to see if cluster assignments have changed
def checktolerance(begin, after):
    #Requires begin and after be same size
    for i in range(0, len(begin)):
        if(begin[i][-1] != after[i][-1]):
            return True #Not converged
    return False #Converged

#Check the value of the distortion function
def distortionfunction(data, clusters):
    distortion = 0
    for point in data:
        for cluster in clusters:
            if(cluster[-1] == point[-1]):
                distortion += (point_to_cluster_distance(point, cluster))**2
    return distortion

#Function to perform kmeans
def kmeans(data, k):
    #Data will be represented in the following way: a list of all points, each point being a list containing the value in each dimension and its final value being an integer identifying the cluster it is assigned to
    data_clustered = list()
    for line in data:
        line.append(0)
        data_clustered.append(line)
    clusters = list()
    point_dimension = len(data_clustered[0]) - 1 #Requires all data points be same size
    #Get maximum data point for scaling random numbers
    max_data_scale = np.amax(np.absolute(np.asarray(data_clustered)))
    #Initialize random clusters
    cluster_id = 0
    rand_clusters = list()
    for x in range(0, k):
        rand_centers = list()
        for y in range(0, point_dimension):
            rand = (2.0*random.random() - 1.0)*max_data_scale
            rand_centers.append(rand)
        rand_centers.append(cluster_id)
        rand_clusters.append(rand_centers)
        cluster_id += 1
    clusters = rand_clusters
    #k-means algorithm
    tolerance = True
    iteration = 0
    distortion_at_iteration = list()
    while(tolerance):
        #Assign each point to its nearest cluster
        begin_data  = copy.deepcopy(data)
        for point in data:
            point_to_cluster_distances = list()
            for cluster in clusters:
                distance = point_to_cluster_distance(point, cluster)
                point_to_cluster_distances.append(distance)
            min_cluster_index = np.argmin(point_to_cluster_distances)
            point[-1] = min_cluster_index
        #Update centroids
        cluster_means = copy.deepcopy(clusters)
        for cluster in cluster_means:
            cluster[:2] = [0] * 2
        cluster_counter = list()
        for cluster in cluster_means:
            cluster_counter.append(0)
        for point in data:
            for cluster in clusters:
                if(point[-1] == cluster[-1]):
                    cluster_counter[point[-1]] += 1.0
                    for i in range(0, len(cluster_means[point[-1]]) - 1):
                        cluster_means[point[-1]][i] += point[i]
        for i in range(0, len(cluster_means)):
            for j in range(0, len(cluster_means[i]) - 1):
                if(cluster_counter[i] != 0):
                    cluster_means[i][j] /= cluster_counter[i]
        for i in range(0, len(clusters)):
            for j in range(0, len(clusters[i])):
                if(cluster_counter[i] != 0):
                    clusters[i][j] = cluster_means[i][j]
        #Check tolerance
        after_data = copy.deepcopy(data)
        tolerance = checktolerance(begin_data, after_data)
        distortion = distortionfunction(after_data, clusters)
        iteration += 1
        distortion_at_iteration.append([distortion, iteration])
    return data, distortion_at_iteration

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
        cleaned_line = filter(None, cleaned_line)
        cleaned_lines.append(cleaned_line)
    converted_lines = list()
    for line in cleaned_lines:
        converted_line = list()
        for element in line:
            element = float(element)
            converted_line.append(element)
        converted_lines.append(converted_line)
    return(converted_lines)

#Here will be the calls to run k-means and produce output
data = loaddata("toydata.txt")
cluster_number = 3

#Run k-means on the toy data for 3 clusters
output, distortion = kmeans(data, cluster_number)

#Plot output dataset
plt.title("Clustering via k-means")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
organized_data = list()

for cluster in range(0, cluster_number):
    organized_data.append([])
for point in data:
    for i in range(0, cluster_number):
        if point[-1] == i:
            organized_data[i].append(point)

f = plt.figure(1)
colors = cm.rainbow(np.linspace(0, 1, cluster_number))
for x, c in zip(range(0, cluster_number), colors):
    label_name = "Cluster number " + str(x)
    plt.scatter([item[0] for item in organized_data[x]], [item[1] for item in organized_data[x]], color=c, marker='o', label=label_name)
leg = plt.legend(loc=2, prop={'size':13}) #Might need to change this if dataset changes
f.show()
f.savefig("kmeans_output.png")

#Run kmeans 20 times
output_20 = list()
distortion_20 = list()
for step in range(0, 20):
    del data[:]
    data = loaddata("toydata.txt")
    output, distortion = kmeans(data, cluster_number)
    output_20.append(output)
    distortion_20.append(distortion)

#Plot distortion function versus iteration number
g = plt.figure(2)
plt.title("Distortion for 20 random k-means runs")
plt.xlabel("Iteration number")
plt.ylabel("Distortion")
colors = cm.rainbow(np.linspace(0, 1, 20))
for x, c in zip(range(0, 20), colors):
    label_name = "k-means run number " + str(x)
    plt.plot([item[1] for item in distortion_20[x]], [item[0] for item in distortion_20[x]], color=c, marker='.', linestyle='-', label=label_name)

#leg = plt.legend(loc=2, prop={'size':6}, ncol=4) #Might need to change this if dataset changes
g.show()
g.savefig("distortion_output.png")
