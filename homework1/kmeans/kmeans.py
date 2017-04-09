#Python code to implement k-means algorithm
#Author: Cody Bezik

import numpy as np
import random
import math
import copy

#Calculates distances between a point and a cluster
def point_to_cluster_distance(point, cluster):
    #print(point, cluster)
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
        #print("Checking convergence")
        #print(begin[i][-1])
        #print(after[i][-1])
        if(begin[i][-1] != after[i][-1]):
            #print("Not converged")
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
    clusters = rand_clusters
    #print(clusters)
    #k-means algorithm
    tolerance = True
    iteration = 0
    distortion_at_iteration = list()
    while(tolerance):
        #Assign each point to its nearest cluster
        begin_data  = copy.deepcopy(data)
        #point_counter = 0
        for point in data:
            point_to_cluster_distances = list()
            #print(point_to_cluster_distances)
            for cluster in clusters:
                #print(cluster)
                distance = point_to_cluster_distance(point, cluster)
                point_to_cluster_distances.append(distance)
            #print(point_to_cluster_distances)
            min_cluster_index = np.argmin(point_to_cluster_distances)
            #print(min_cluster_index)
            #temp = data[point_counter][-1]
            #print(data[point_counter][-1])
            point[-1] = min_cluster_index
            #print(temp, data[point_counter][-1])
            #point_counter += 1
            #print(point)
        #Update centroids
        cluster_means = copy.deepcopy(clusters)
        for cluster in cluster_means:
            cluster[:2] = [0] * 2
        #print(cluster_means)
        cluster_counter = list()
        for cluster in cluster_means:
            cluster_counter.append(0)
        #print(cluster_counter)
        for point in data:
            for cluster in clusters:
                if(point[-1] == cluster[-1]):
                    cluster_counter[point[-1]] += 1.0
                    for i in range(0, len(cluster_means[point[-1]]) - 1):
                        cluster_means[point[-1]][i] += point[i]
                    #print("Count this point in mean")
        for i in range(0, len(cluster_means)):
            for j in range(0, len(cluster_means[i]) - 1):
                if(cluster_counter[i] != 0):
                    cluster_means[i][j] /= cluster_counter[i]
                else:
                    print("No points assigned to cluster " + str(i))
        for i in range(0, len(clusters)):
            for j in range(0, len(clusters[i])):
                if(cluster_counter[i] != 0):
                    clusters[i][j] = cluster_means[i][j]
        #print(cluster_means)
        #print(clusters)
        #Check tolerance
        after_data = copy.deepcopy(data)
        #print(begin_data)
        #print(after_data)
        tolerance = checktolerance(begin_data, after_data)
        distortion = distortionfunction(after_data, clusters)
        #print(tolerance)
        #tolerance = False
        iteration += 1
        #data = copy.deepcopy(after_data)
        print("Iteration " + str(iteration) + " Distortion = " + str(distortion))
        distortion_at_iteration.append([distortion, iteration])
    print("Converged!")
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
output, distortion = kmeans(data, 3)
#print(output)
#Plot output dataset

#Run kmeans 20 times
output_20 = list()
distortion_20 = list()
for step in range(0, 20):
    del data[:]
    #print(data)
    data = loaddata("toydata.txt")
    output, distortion = kmeans(data, 3)
    output_20.append(output)
    distortion_20.append(distortion)

print(distortion_20)
