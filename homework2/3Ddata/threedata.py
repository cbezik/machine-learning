import numpy as np
import matplotlib.pyplot as plt

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

#Generate eigenvector matrix in specified number of dimensions
def geteigenmatrix(pairs, dimension):
    if(dimension == 2):
        matrix_w = np.hstack((pairs[0][1].reshape(3, 1), pairs[1][1].reshape(3,1)))
    return matrix_w

#Project data onto subspace
def projectdata(matrix, data):
    transformed = matrix.T.dot(data)
    return(transformed)


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

#print(pca_output)
#Tag points with appropriate color
for i, point in enumerate(data):
    #print(int(point[3]))
    if(point[3] == 1):
        plt.plot([pca_output[0,i]], [pca_output[1,i]], color="green", marker='o')
    if(point[3] == 2):
        plt.plot([pca_output[0,i]], [pca_output[1,i]], color="yellow", marker='o')
    if(point[3] == 3):
        plt.plot([pca_output[0,i]], [pca_output[1,i]], color="blue", marker='o')
    if(point[3] == 4):
        plt.plot([pca_output[0,i]], [pca_output[1,i]], color="red", marker='o')

f = plt.figure(1)

#plt.scatter(pca_output[0], pca_output[1], color="red", marker='o')

"""colors = cm.rainbow(np.linspace(0, 1, cluster_number))
for x, c in zip(range(0, cluster_number), colors):
    label_name = "Cluster number " + str(x)
    plt.scatter([item[0] for item in organized_data[x]], [item[1] for item in organized_data[x]], color=c, marker='o', label=label_name)
leg = plt.legend(loc=2, prop={'size':13}) #Might need to change this if dataset changes
"""

f.show()
f.savefig("pca_output.png")
