import numpy as np


'''
Code out kmeans clustering algo
  1. Randomly select k observations to serve as initial cluster centroid
  2. Assign every observation to the centroid it is closest to (Euclidean dist)
  3. Compute new cluster centroids
  4. Repeat 2-3 until convergence (when cluster centroids are same as
     last iter)
'''


def kmeans(dat, K, seed=0):
    '''
    Input:
        dat = N x M numerical matrix (numpy 2D array)
        K = the number of desired clusters
    Output:
        clust_centers: an K x M matrix representing cluster centers
        clust_assignments: 1D array of length N of cluster assignments
    '''
    np.random.seed(seed)
    N, M = dat.shape

    # Randomly choose initial centroids
    centroids_last = np.array([dat[i] for i in np.random.randint(0, N, K)])

    converged = False
    num_iter = 0

    '''Compute Euclidean distance for 1 cluster centroid
    dist = []
    for i in range(N):
        diff = dat[i] - centroids[0]
        dist.append(np.sqrt(diff.dot(diff)))
    '''
    while not converged:
        # Compute Euclidean distance for obs vs all centroids
        # print(centroids_last)
        dist = []
        for i in range(K):
            diff = dat - centroids_last[i]
            dist.append(np.sqrt((diff * diff).sum(axis=1)))

        dist = np.array(dist).T
        clust_nums = np.argmin(dist, axis=1)

        centroids_next = np.array([dat[clust_nums == i].mean(axis=0)
                                   for i in range(K)])

        # If no change, then kmeans has converged
        converged = (centroids_last == centroids_next).all()
        centroids_last = centroids_next
        num_iter += 1

    print('Took {} iterations until convergence'.format(num_iter))

    return(centroids_next, clust_nums)
