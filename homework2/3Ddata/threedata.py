import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
#from sklearn.neighbors import NearestNeighbors

#This code will import the dataset 3Ddata.txt and perform the following:
    #PCA reducing its dimensionality from 3 to 2

#Reads in data
def loaddata(filename):
    with open(filename) as f:
        lines = f.readlines()
    lines = [i.split(' ') for i in lines]
    #lines = [i.split('\n') for i in lines]
    for line in lines:
        line[-1] = line[-1].strip('\n')
        #line = [float(i) for i in line]
    #lines = [float(x) for line in lines for x in line]
    lines=np.asarray(lines)
    lines=lines.astype(float)
    #for i, line in enumerate(lines):
        #lines[i][-1] = int(line[-1])
    return lines

#Returns the mean vector of the dataset
def getmeanvector(data):
    mean_x = np.mean(data[:, 0])
    mean_y = np.mean(data[:, 1])
    mean_z = np.mean(data[:, 2])
    mean_vector = np.array([[mean_x], [mean_y], [mean_z]])
    return mean_vector

#Returns the covariance matrix of the dataset
def getcovarmatrix(data):
    cov_mat = np.cov([data[:,0], data[:,1], data[:, 2]])
    return cov_mat

#Return eigenvalue, eigenvector of input matrix
def geteigens(matrix):
    eig_val, eig_vec = np.linalg.eig(matrix)
    return eig_val, eig_vec

#Sort eigenvectors in order of decreasing eigenvalue
def sorteigens(eig_val, eig_vec):
    #Arrange pairs in tuple
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
    #Sort pairs
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    return eig_pairs

#Generates matrix of two leading eigenvectors with specified number of dimensions
def geteigenmatrix(pairs, dimension):
    rows = dimension + 1
    columns = 1
    matrix_w = np.hstack((pairs[0][1].reshape(rows, columns), pairs[1][1].reshape(rows, columns)))
    return matrix_w

#Diagonal matrix containing 2 leading eigenvalues on the diagonals
def geteigenvaluematrix(pairs):
    s = (2,2)
    matrix_l = np.zeros(s)
    matrix_l[0][0] = pairs[0][0]
    matrix_l[1][1] = pairs[1][0]
    return matrix_l

#Project data onto subspace
def projectdata(matrix, data):
    transformed = matrix.T.dot(data)
    return(transformed)

#Floyd's algorithm on a matrix, currently requires nxn matrix
def floyds(matrix):
    dimension = matrix[0].size
    output = matrix
    for k in range(0, dimension):
        for i in range(0, dimension):
            for j in range(0, dimension):
                dij = matrix[i][j]
                dik = matrix[i][k]
                dkj = matrix[k][j]
                replacement_val = min(dij, dik + dkj)
                output[i][j] = replacement_val
    return output

#First import the dataset
data = loaddata("3Ddata.txt")

#Do PCA
pca_mean = getmeanvector(data)
data[:, 0] = data[:, 0] - pca_mean[0]
data[:, 1] = data[:, 1] - pca_mean[1]
data[:, 2] = data[:, 2] - pca_mean[2]
pca_covar = getcovarmatrix(data)
eig_val_cov, eig_vec_cov = geteigens(pca_covar)
eig_pairs = sorteigens(eig_val_cov, eig_vec_cov)
eigen_matrix = geteigenmatrix(eig_pairs, 2)
#Reorganize data
reorganized_data = np.vstack((data[:,0], data[:,1],data[:,2]))

pca_output = projectdata(eigen_matrix, reorganized_data)

#Plot PCA results


#Tag points with appropriate color
for i, point in enumerate(data):
    if(point[3] == 1):
        plt.plot([pca_output[0,i]], [pca_output[1,i]], color="green", marker='o')
    if(point[3] == 2):
        plt.plot([pca_output[0,i]], [pca_output[1,i]], color="yellow", marker='o')
    if(point[3] == 3):
        plt.plot([pca_output[0,i]], [pca_output[1,i]], color="blue", marker='o')
    if(point[3] == 4):
        plt.plot([pca_output[0,i]], [pca_output[1,i]], color="red", marker='o')

f = plt.figure(1)

"""colors = cm.rainbow(np.linspace(0, 1, cluster_number))
for x, c in zip(range(0, cluster_number), colors):
    label_name = "Cluster number " + str(x)
    plt.scatter([item[0] for item in organized_data[x]], [item[1] for item in organized_data[x]], color=c, marker='o', label=label_name)
leg = plt.legend(loc=2, prop={'size':13}) #Might need to change this if dataset changes
"""

f.show()
f.savefig("pca_output.png")

#Do Isomap
#Re-import the dataset
data = loaddata("3Ddata.txt")

#Compute distances between every point (not sure I need this anymore)
reorganized_data = np.vstack((data[:,0], data[:,1],data[:,2]))
Y = spatial.distance.cdist(reorganized_data.T, reorganized_data.T, 'euclidean') #Accepts input in the form of m x n array, m = number of observations, n = number of dimensions

#Find k-nearest neighbors
NN_tree = spatial.cKDTree(reorganized_data.T)
"""
for point in reorganized_data.T:
    k_nns = NN_tree.query(point, k=10) #k_nns is a tuple containing a list of the nearest neighbor distances and the associated index of the nearest points
"""

for i in range(0, Y[0].size):
    k_nns = NN_tree.query(reorganized_data.T[i], k=11, p=2)
    for j in range(0, Y[0].size):
        if(j == i):
            Y[i][j] = 0
        elif(j in k_nns[1]):
            continue
        else:
            Y[i][j] = float("inf")

#Do Floyd's algorithm
shortest = floyds(Y)

#Create matrix of squared distances
squared_shortest = np.square(shortest)
#Create centering matrix
centering = np.zeros((squared_shortest[0].size, squared_shortest[0].size))
np.fill_diagonal(centering, 1.0)
centering = centering - (1.0 / squared_shortest[0].size)
tau = reduce(np.dot, [centering, squared_shortest, centering])
tau = -0.5*tau
eig_val_cov, eig_vec_cov = geteigens(tau)
eig_pairs = sorteigens(eig_val_cov, eig_vec_cov)
eigen_matrix = geteigenmatrix(eig_pairs, squared_shortest[0].size - 1)
lambda_matrix = geteigenvaluematrix(eig_pairs)
isomap_output = np.dot(np.real(eigen_matrix), np.sqrt(lambda_matrix))

#Plot Isomap results

g = plt.figure(2)

#Tag points with appropriate color
for i, point in enumerate(data):
    if(int(point[3]) == 1):
        plt.plot([isomap_output[i,0]], [isomap_output[i,1]], color="green", marker='o')
    if(int(point[3]) == 2):
        plt.plot([isomap_output[i,0]], [isomap_output[i,1]], color="yellow", marker='o')
    if(int(point[3]) == 3):
        plt.plot([isomap_output[i,0]], [isomap_output[i,1]], color="blue", marker='o')
    if(int(point[3]) == 4):
        plt.plot([isomap_output[i,0]], [isomap_output[i,1]], color="red", marker='o')

g.show()
g.savefig("isomap_output.png")

#np.set_printoptions(threshold='nan')
