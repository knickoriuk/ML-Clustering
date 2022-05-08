import numpy as np
from functools import partial
from scipy.stats import multivariate_normal

class ClusterModel:
    '''
    Attributes:
        K : int
            Number of clusters for the model
        D : int
            Dimensionality of the data
        model_type : string, optional
            The model used for clustering. Choose either "kmeans" or "GMM".
        init_algorithm : string, optional
            Algorithm used to initialize group centers, "max_dist" or "random".
        labels : numpy.ndarray (shape: (N, 1))
            Cluster label for each data point.
        centers : numpy.ndarray (shape: (K, D))
            A D-dimensional center point of each cluster in K.
        covariances : numpy.ndarray (shape: (K, D, D))
            For GMM models, a DxD covariance matrix of each cluster in K.
        mixture_proportions : numpy.ndarray (shape: (K, 1))
            For GMM models, a vector of K proportions for each gaussian 
            distribution, which total 1.
        probability_matrix : numpy.ndarray (shape: (N, K))
            For GMM models, a matrix of probabilities that each point i 
            belongs to each gaussian distribution k.
    '''
    
    def __init__(self, K, model_type="kmeans", init_algorithm="max_dist"):
        '''
        Parameters
        ----------
        K : int
            Number of clusters to group data into.
        model_type : string, optional
            The model used for clustering. Choose either "kmeans" or "GMM" for
            k-means or Gaussian mixture models, respectively.
        init_algorithm : string, optional
            Algorithm used to initialize group centers. Choose "max_dist" to 
            initialize centers with maximized distance between them, or 
            choose "random" to select random points to be centers.
        '''
        assert K > 1, f"There must be at least 2 clusters. Got: {self.K}"
        assert model_type in ["kmeans", "GMM"], "Model type must be either 'kmeans' or 'GMM'."
        assert init_algorithm in ["max_dist", "random"], "Initialization algorithm must be either 'max_dist' or 'random'."
        
        self.K = K
        self.model_type = model_type
        self.init_algorithm = init_algorithm
    
    def train(self, X, max_iterations=1000):
        '''
        Runs the chosen model type's algorithm and clusters the given X data,
        sorting each point into one of K groups. 

        Parameters
        ----------
        X : numpy.ndarray (shape: (N, D))
            The array of data to be clustered.
        max_iterations : int, optional
            The number times the algorithm loops before quitting.
        '''
        assert max_iterations > 0, "'max_iterations' must be a positive value"
        (N, self.D) = X.shape
        
        ############## Initialize Centers Using Chosen Algorithm ##############
        if self.init_algorithm == "random":
            self.centers = X[np.random.randint(X.shape[0], size=self.K)]
            
        elif self.init_algorithm == "max_dist":
            self.centers = np.empty(shape=(self.K, self.D))
            
            # First center is chosen randomly:
            self.centers[0] = X[np.random.randint(N)]
            
            # For each of the remaining clusters, 
            # Find the data point farthest from all existing centroids
            for k in range(self.K-1):
                distances = np.array([])
                
                # Compare with every data point:
                for i in range(N):
                    point = X[i, :]
                    dist_1 = np.inf
                    
                    # Find distance to closest center
                    for j in range(self.K):
                        dist_2 = np.linalg.norm(point - self.centers[j])
                        dist_1 = min(dist_1, dist_2)
                    distances = np.append(distances, dist_1)
                
                # Pick the datapoint with farthest distance from all other centers
                self.centers[k+1] = X[np.argmax(distances), :]
        
        ########### Run Respective Model's Algorithm for Clustering ###########
        # 1. K-MEANS
        # initialize arrays for labels and distances
        labels = np.empty(shape=(N, 1), dtype=np.compat.long)
        distances = np.empty(shape=(N, self.K))
        
        for _ in range(max_iterations):
            old_labels = np.copy(labels)
            
            ##### UPDATE LABELS
            # Find all distances between datapoints and each centroid
            for i in range(N):
                for k in range(self.K):
                    # Norm of ith data point from kth cluster center
                    dist_to_k = np.linalg.norm(X[i,:] - self.centers[k])
                    distances[i, k] = dist_to_k
                
            # Set each datapoint's label to the index with the least distance
            labels = np.argmin(distances, axis=1)
            
            ##### UPDATE CENTERS
            # Set kth cluster center as the mean of all points labelled k
            for k in range(self.K):
                self.centers[k,:] = np.mean(X[labels==k, :], axis=0)
            labels = np.reshape(labels, (N,1))

            ##### CONVERGENCE CHECK
            if np.allclose(old_labels, labels):
                break
    
        # 2. GAUSSIAN MIXTURE MODEL
        if self.model_type == "GMM":
        # Use result of k-means as initialization for GMM centers
            
            # Initialize covariance matrices and mixture proportions
            self.covariances = np.tile(np.eye(self.D), reps=(self.K, 1, 1))
            self.mixture_proportions = np.ones(shape=(self.K, 1)) / self.K
            labels = np.empty(shape=(N, 1), dtype=np.compat.long)
            
            # Define functions e_step and m_step using partials
            e_step = partial(self._e_step, X=X)
            m_step = partial(self._m_step, X=X)
    
            for _ in range(max_iterations):
                old_labels = labels
                
                ##### E-STEP
                e_step()
                # Reassign labels
                labels = np.argmax(self.probability_matrix, axis=1).reshape((N, 1))
    
                ##### CONVERGENCE CHECK
                if np.allclose(old_labels, labels):
                    break
    
                ##### M-STEP
                m_step()

        self.labels = np.copy(labels)

    def _e_step(self, X):
        '''
        Performs the E-step of the EM Algorithm for GMM models. Determines a 
        probability matrix for the probability that data point i belongs to
        cluster k, given current gaussian parameters.

        Parameters
        ----------
        X : numpy.ndarray (shape: (N, D))
            The array of data to be clustered.
        '''
        N = X.shape[0]
        probability_matrix = np.empty(shape=(N, self.K))

        for k in range(self.K):
                
            pi_k = self.mixture_proportions[k]  # shape: (1)
            mu_k = self.centers[k]              # shape: (D, 1)
            sigma_k = self.covariances[k]       # shape: (D, D)
            sigma_k += np.eye(self.D)*1e-6 # Prevents singularities
            
            likelihood = multivariate_normal(mu_k, sigma_k).pdf(X)               
            probability_matrix[:, k] = pi_k * likelihood
        
        self.probability_matrix = probability_matrix / probability_matrix.sum(axis=1, keepdims=1)
        return
        
    def _m_step(self, X):
        '''
        Performs the E-step of the EM Algorithm for GMM models. Determines 
        parameters for the Gaussian distributions that maximize likelihood of
        generating the points assigned to them.

        Parameters
        ----------
        X : numpy.ndarray (shape: (N, D))
            The array of data to be clustered.
        '''
        N = X.shape[0]
        
        # Total probability assigned to each cluster:
        prob_k = np.sum(self.probability_matrix, axis=0)
        
        # Update Mixture Proportions:
        self.mixture_proportions = np.reshape(prob_k / N, (self.K,1))
        
        # Update Means:
        weighted_sum = np.dot(self.probability_matrix.T, X)
        self.centers = weighted_sum / prob_k.reshape(-1, 1)
        
        # Update Covariances:
        self.covariances = np.empty(shape=(self.K, self.D, self.D))
        for k in range(self.K):
            # residual = (x_i - mu_k)
            residual = (X - self.centers[k])
            weighted_sum = np.dot(self.probability_matrix[:, k] * residual.T, residual)
            self.covariances[k] = (weighted_sum / prob_k[k])
        return