import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from scipy.linalg import solve

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

#Generates matrix of two bottom eigenvectors with specified number of dimensions
def getreverseeigenmatrix(pairs, dimension):
    rows = dimension + 1
    columns = 1
    size_pairs = len(pairs)
    matrix_w = np.hstack((pairs[size_pairs - 2][1].reshape(rows, columns), pairs[size_pairs - 3][1].reshape(rows, columns)))
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

#Run PCA
def PCA():
    print("Doing PCA...")
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

#Run Isomap
def Isomap():
    print("Doing Isomap..")
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

#Do LLE
def LLE():
    print("Doing LLE...")
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
    nearest_neighbors = 10 #User input
    nearest_neighbors += 1
    data_dimension = data[0].size - 1
    #print(data_dimension)
    #Build weight matrix
    number_points = reorganized_data[0].size
    #print(number_points)
    weights = np.zeros((number_points, number_points))
    #print(weights)
    for i in range(0, number_points):
        k_nns = NN_tree.query(reorganized_data.T[i], k=nearest_neighbors, p=2)
        #print(k_nns)
        Xi = reorganized_data.T[k_nns[1][0]]
        Xi = np.reshape(Xi, (3, 1))
        #print(Xi)
        #Matrix Z ultimately consists of nearest neighbors minus data point
        #Z = np.delete(reorganized_data, i, axis=1)
        #print(Z)
        Z = np.zeros((data_dimension, nearest_neighbors - 1))
        for j in range(1, nearest_neighbors):
            #print(j)
            #print(k_nns)
            #print(reorganized_data.T[k_nns[1][j]])
            Z[:,j - 1] = reorganized_data.T[k_nns[1][j]]
        #print(Z)
        Z = Z - Xi
        C = np.cov(Z.T)
        #Regularize covariance matrix to get full rank
        eps = 1e-2*np.trace(C) #/ (nearest_neighbors - 1)
        #C = C+eps*np.identity(number_points - 1)
        C = C + eps*np.identity(nearest_neighbors - 1)
        #C_inverse = np.linalg.inv(C) #Likely not strictly the most efficient but okay for the small size we have here
        ones = np.ones((nearest_neighbors - 1, 1))
        w = solve(C, ones)
        #print(Z)
        #print(C)
        #print(C_inverse)
        #sum_c_inverse = np.sum(C_inverse)
        sum_w = np.sum(w)
        #print(sum_c_inverse)
        #print(k_nns[1][1:])
        #print(w)
        weight_getter = 0
        #sorted_knns = np.sort(k_nns[1][1:])
        sorted_knns = k_nns[1][1:]
        #print(sorted_knns)
        for j in sorted_knns:
            """
            if(j > i):
                weights[i][j] = np.sum(C_inverse[j - 1]) / sum_c_inverse
            else:
                weights[i][j] = np.sum(C_inverse[j]) / sum_c_inverse
            """
            weights[i][j] = w[weight_getter] / sum_w
            weight_getter += 1
    #print(weights)
    M = np.cov((np.identity(number_points) - weights).T)
    eig_val, eig_vec = geteigens(M)
    #print(eig_val)
    #print(eig_vec)
    eig_pairs = sorteigens(eig_val, eig_vec)
    #print(eig_pairs)
    eigen_matrix = getreverseeigenmatrix(eig_pairs, number_points - 1)
    #eigen_matrix /= eig_pairs[number_points - 1][1][0] #This is a hack but the bottom eigenvector is supposed to be a matrix of all ones but instead it's some constant less than one, so I'm taking that constant and rescaling the other eigenvectors
    #print(eigen_matrix)
    #print(len(eig_pairs))
    #print(eig_pairs[499][1])
    #print(eig_pairs[499][0])
    #print(eig_pairs[498][0])
    #print(eig_pairs[497][0])

    lle_output = eigen_matrix

    #Plot LLE results

    h = plt.figure(3)

    #Tag points with appropriate color
    for i, point in enumerate(data):
        if(int(point[3]) == 1):
            plt.plot([lle_output[i,0]], [lle_output[i,1]], color="green", marker='o')
        if(int(point[3]) == 2):
            plt.plot([lle_output[i,0]], [lle_output[i,1]], color="yellow", marker='o')
        if(int(point[3]) == 3):
            plt.plot([lle_output[i,0]], [lle_output[i,1]], color="blue", marker='o')
        if(int(point[3]) == 4):
            plt.plot([lle_output[i,0]], [lle_output[i,1]], color="red", marker='o')

    h.show()
    h.savefig("lle_output.png")


#Run various methods
def main():
    #PCA()
    #Isomap()
    LLE()



np.set_printoptions(threshold='nan')
main()
