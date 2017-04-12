#Python code to implement mixture of Gaussians EM algorithm
#Author: Cody Bezik

import numpy as np
import random
import math
import copy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from scipy.stats import multivariate_normal
import sys
import operator

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

#Function to perform kmeans++
def kmeanspp(data, k):
    #Data will be represented in the following way: a list of all points, each point being a list containing the value in each dimension and its final value being an integer identifying the cluster it is assigned to
    data_clustered = list()
    for line in data:
        line.append(0)
        data_clustered.append(line)
    clusters = list()
    point_dimension = len(data_clustered[0]) - 1 #Requires all data points be same size
    #Get maximum data point for scaling random numbers
    max_data_scale = np.amax(np.absolute(np.asarray(data_clustered)))
    #Initialize clusters with k-means++ algorithm
    pp_clusters = list()
    #Choose a random center from the data points
    rand_id = int(math.ceil(random.random()*len(data)) - 1)
    center = data[rand_id]
    pp_clusters.append(center)
    for x in range(1, k):
        pp_distances = list()
        #Compute D(x)^2
        for point in data:
            temp_distances = list()
            for cluster in range(0, x):
                distance = point_to_cluster_distance(point, pp_clusters[cluster])
                temp_distances.append(distance)
            min_dist = np.amin(temp_distances)
            pp_distances.append(min_dist**2)
        #Calculate probability
        sum_distances = np.sum(np.asarray(pp_distances))
        probs = pp_distances / sum_distances
        #Choose new center according to new probability distribution
        center_id = np.random.choice(np.asarray(range(0, len(data))), p = np.asarray(probs))
        center = data[center_id]
        center[-1] = x
        #Add center to list of clusters
        pp_clusters.append(center)
    clusters = pp_clusters
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
    return data, distortion_at_iteration, clusters, point_dimension

def log_likelihood(data, means, covars, weights):
    likelihood = 0
    for i in range(0, len(data)):
        for j in range(0, len(means)):
            N = multivariate_normal.pdf(data[i][:-1], mean=means[j], cov=covars[j])
            interior = weights[j]*N
            likelihood += interior
    return likelihood

def em_tolerance(before, after):
    diff = abs(after - before)
    if(diff >= 1e-2):
        return True
    else:
        return False

def gaussmixem(data, k):
    #Data will be represented in the following way: a list of all points, each point being an array containing the point's coordinates and its final value being a list of probabilities that it belongs to each cluster
    clean_data = copy.deepcopy(data)
    output = list()
    likelihood = list()
    iteration = 0

    #Run k-means++ on the data to initialize
    output_kmeans, distortion_kmeans, clusters_kmeans, point_dimensionality = kmeanspp(data, k)

    data = copy.deepcopy(clean_data)
    for point in data:
        point.append(range(0, k))

    #Lists to hold initial values
    #Initialize means
    initial_gauss_means = copy.deepcopy(clusters_kmeans)
    for mean in initial_gauss_means:
        mean.pop()
    #Initialize covars
    initial_gauss_covars = list()
    for cluster in range(0, k):
        initial_gauss_covars.append([])
        for dimension in range(0, point_dimensionality):
            initial_gauss_covars[cluster].append([])
            for row in range(0, point_dimensionality):
                initial_gauss_covars[cluster][dimension].append(0)
    point_counter = list()
    for cluster in range(0, k):
        point_counter.append(0)
    for point in output_kmeans:
        for i in range(0, len(initial_gauss_means)):
            if point[-1] == i:
                point_counter[i] += 1.0
                temp_point = np.asarray(point[:-1])
                temp_mean = np.asarray(initial_gauss_means[i])
                point_minus_mean = temp_point - temp_mean
                product = np.outer(point_minus_mean, point_minus_mean) #Necessitates data points be represented as one dimensional arrays
                initial_gauss_covars[i] += product
    for i in range(0 , len(initial_gauss_covars)):
        for j in range(0, len(initial_gauss_covars[i])):
            for l in range(0, len(initial_gauss_covars[i][j])):
                initial_gauss_covars[i][j][l] /= point_counter[i]
    #Initialize weights
    initial_gauss_weights = list()
    for i in range(0, k):
        initial_gauss_weights.append(point_counter[i] / len(data))
    #Set up for algorithm
    tolerance = True
    gauss_means = copy.deepcopy(initial_gauss_means)
    gauss_covars = copy.deepcopy(initial_gauss_covars)
    gauss_weights = copy.deepcopy(initial_gauss_weights)
    while(tolerance):
        #E-step - update the assignments
        begin_likelihood = log_likelihood(data, gauss_means, gauss_covars, gauss_weights)
        for point in data:
            normals = list()
            for i in range(0, k):
                N = multivariate_normal.pdf(point[:-1], mean = gauss_means[i], cov=gauss_covars[i])
                normals.append(N)
            sum_weights_norms = 0
            for i in range(0, k):
                sum_weights_norms += gauss_weights[i]*normals[i]
            for i in range(0, len(point[-1])):
                pij = (gauss_weights[i]*normals[i]) / sum_weights_norms
                point[-1][i] = pij
        #M-step - updating weights, means, covars
        pij_sum = list()
        for i in range(0, k):
            pij_sum.append(0)
        #For each j, sum pijs
        for point in data:
            for i in range(0, len(point[-1])):
                pij_sum[i] += point[-1][i]
        n = len(data)
        for j in range(0, k):
            gauss_weights[j] = pij_sum[j] / float(n)
        pij_xi_sum = list()
        #Below this line doesn't properly support beyond 2D points
        for i in range(0, k):
            pij_xi_sum.append([0.0, 0.0])
        pij_xi_sum = np.asarray(pij_xi_sum)
        for point in data:
            for i in range(0, len(point[-1])):
                temp_point = np.asarray(point[:-1])*point[-1][i]
                pij_xi_sum[i] = pij_xi_sum[i] + temp_point
        for j in range(0, k):
            gauss_means[j] = pij_xi_sum[j] / pij_sum[j]
        pij_sum_x_mu = list()
        for i in range (0, k):
            pij_sum_x_mu.append([[0.0,0.0], [0.0,0.0]])
        pij_sum_x_mu = np.asarray(pij_sum_x_mu)
        for point in data:
            for i in range(0, len(point[-1])):
                temp_point = np.asarray(point[:-1])
                temp_mean = np.asarray(gauss_means[i])
                point_minus_mean = temp_point - temp_mean
                product = np.outer(point_minus_mean, point_minus_mean)*point[-1][i] #Necessitates data points be represented as one dimensional arrays
                pij_sum_x_mu[i] = np.add(pij_sum_x_mu[i], product, casting='no')
        for j in range(0, k):
            gauss_covars[j] = pij_sum_x_mu[j] / pij_sum[j]
        after_likelihood = log_likelihood(data, gauss_means, gauss_covars, gauss_weights)
        tolerance = em_tolerance(begin_likelihood, after_likelihood)
        iteration += 1
        likelihood.append([after_likelihood, iteration])
    return data, likelihood


#Import dataset and set parameters
data = loaddata("toydata.txt")
cluster_number = 3
output, output_likelihood = gaussmixem(data, cluster_number)

for point in output:
    index = np.argmax(point[-1])
    point.pop()
    point.append(index)

#Plot output dataset
plt.title("Clustering via Mixture of Gaussians EM")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
organized_data = list()

for cluster in range(0, cluster_number):
    organized_data.append([])
for point in output:
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
f.savefig("gauss_output.png")
