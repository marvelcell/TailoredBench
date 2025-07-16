"""K-medoids clustering"""

# Authors: Timo Erkkilä <timo.erkkila@gmail.com>
#          Antti Lehmussola <antti.lehmussola@gmail.com>
#          Kornel Kiełczewski <kornel.mail@gmail.com>
#          Zane Dufour <zane.dufour@gmail.com>
# License: BSD 3 clause

import warnings

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import (
    pairwise_distances,
    pairwise_distances_argmin,
)
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import ConvergenceWarning

# cython implementation of steps in PAM algorithm.
from ._k_medoids_helper import _compute_optimal_swap, _build

def custom_pairwise_distances_argmin(X, Y):
    """
    Using custom distance metric, for each sample in X, find the index of the nearest sample in Y.
    
    Parameters:
    X : ndarray, shape (n_samples_X, n_features)
        The sample set to find nearest neighbors for.
    Y : ndarray, shape (n_samples_Y, n_features)
        The candidate nearest neighbor sample set.
    
    Returns:
    indices : ndarray, shape (n_samples_X,)
        indices[i] is the sample index in Y that minimizes the custom distance.
    """
    # Calculate norms (magnitudes) of X and Y
    norms_X = np.linalg.norm(X, axis=1)  # shape (n_samples_X,)
    norms_Y = np.linalg.norm(Y, axis=1)  # shape (n_samples_Y,)
    
    # Calculate Euclidean distance matrix between all vectors (numerator part)
    numerator = cdist(X, Y, metric='euclidean')  # shape (n_samples_X, n_samples_Y)
    
    # Calculate sum of norms matrix (denominator part)
    denominator = norms_X[:, np.newaxis] + norms_Y[np.newaxis, :]  # shape (n_samples_X, n_samples_Y)
    
    # Prevent division by zero
    epsilon = np.finfo(float).eps  # machine epsilon (smallest positive number)
    denominator = np.where(denominator == 0, epsilon, denominator)
    
    # Calculate custom distance matrix D
    D = numerator / denominator
    
    # For each sample in X, find the index of the sample in Y that minimizes distance
    indices = np.argmin(D, axis=1)  # shape (n_samples_X,)
    
    return indices

def compute_similarity_matrix(X, Y=None):
    """
    Compute similarity matrix based on specified distance.
    
    Parameters:
    X : ndarray, shape (n_samples_1, n_features)
        Each row represents a vector representation of an element.
    Y : ndarray, shape (n_samples_2, n_features), optional
        If provided, will compute similarity matrix between X and Y.
        If not provided, will compute similarity matrix of X with itself.
    
    Returns:
    D : ndarray, shape (n_samples_1, n_samples_2) or (n_samples_1, n_samples_1)
        Similarity matrix between elements.
    """
    # If Y is not provided, set Y = X
    if Y is None:
        Y = X
    # Calculate norms (magnitudes) of X and Y
    norms_X = np.linalg.norm(X, axis=1)  # shape (n_samples_1,)
    norms_Y = np.linalg.norm(Y, axis=1)  # shape (n_samples_2,)
    
    # Calculate Euclidean distance matrix between all vectors (numerator part)
    numerator = cdist(X, Y, metric='euclidean')  # shape (n_samples_1, n_samples_2)
    
    # Calculate sum of norms matrix (denominator part)
    denominator = norms_X[:, np.newaxis] + norms_Y[np.newaxis, :]  # shape (n_samples_1, n_samples_2)
    
    # Prevent division by zero (in case of zero vectors)
    epsilon = np.finfo(float).eps  # machine epsilon (smallest positive number)
    denominator = np.where(denominator == 0, epsilon, denominator)
    
    # Calculate similarity matrix
    D = numerator / denominator

    return D


def _compute_inertia(distances):
    """Compute inertia of new samples. Inertia is defined as the sum of the
    sample distances to closest cluster centers.

    Parameters
    ----------
    distances : {array-like, sparse matrix}, shape=(n_samples, n_clusters)
        Distances to cluster centers.

    Returns
    -------
    Sum of sample distances to closest cluster centers.
    """

    # Define inertia as the sum of the sample-distances
    # to closest cluster centers
    inertia = np.sum(np.min(distances, axis=1))

    return inertia


class KMedoids(BaseEstimator, ClusterMixin, TransformerMixin):
    """k-medoids clustering.

    Read more in the :ref:`User Guide <k_medoids>`.

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of medoids to
        generate.

    metric : string, or callable, optional, default: 'euclidean'
        What distance metric to use. See :func:metrics.pairwise_distances
        metric can be 'precomputed', the user must then feed the fit method
        with a precomputed kernel matrix and not the design matrix X.

    method : {'alternate', 'pam'}, default: 'alternate'
        Which algorithm to use. 'alternate' is faster while 'pam' is more accurate.

    init : {'random', 'heuristic', 'k-medoids++', 'build'}, or array-like of shape
        (n_clusters, n_features), optional, default: 'heuristic'
        Specify medoid initialization method. 'random' selects n_clusters
        elements from the dataset. 'heuristic' picks the n_clusters points
        with the smallest sum distance to every other point. 'k-medoids++'
        follows an approach based on k-means++_, and in general, gives initial
        medoids which are more separated than those generated by the other methods.
        'build' is a greedy initialization of the medoids used in the original PAM
        algorithm. Often 'build' is more efficient but slower than other
        initializations on big datasets and it is also very non-robust,
        if there are outliers in the dataset, use another initialization.
        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        .. _k-means++: https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf

    max_iter : int, optional, default : 300
        Specify the maximum number of iterations when fitting. It can be zero in
        which case only the initialization is computed which may be suitable for
        large datasets when the initialization is sufficiently efficient
        (i.e. for 'build' init).

    random_state : int, RandomState instance or None, optional
        Specify random state for the random number generator. Used to
        initialise medoids when init='random'.

    Attributes
    ----------
    cluster_centers_ : array, shape = (n_clusters, n_features)
            or None if metric == 'precomputed'
        Cluster centers, i.e. medoids (elements from the original dataset)

    medoid_indices_ : array, shape = (n_clusters,)
        The indices of the medoid rows in X

    labels_ : array, shape = (n_samples,)
        Labels of each point

    inertia_ : float
        Sum of distances of samples to their closest cluster center.

    Examples
    --------
    >>> from sklearn_extra.cluster import KMedoids
    >>> import numpy as np

    >>> X = np.asarray([[1, 2], [1, 4], [1, 0],
    ...                 [4, 2], [4, 4], [4, 0]])
    >>> kmedoids = KMedoids(n_clusters=2, random_state=0).fit(X)
    >>> kmedoids.labels_
    array([0, 0, 0, 1, 1, 1])
    >>> kmedoids.predict([[0,0], [4,4]])
    array([0, 1])
    >>> kmedoids.cluster_centers_
    array([[1., 2.],
           [4., 2.]])
    >>> kmedoids.inertia_
    8.0

    See scikit-learn-extra/examples/plot_kmedoids_digits.py for examples
    of KMedoids with various distance metrics.

    References
    ----------
    Maranzana, F.E., 1963. On the location of supply points to minimize
      transportation costs. IBM Systems Journal, 2(2), pp.129-135.
    Park, H.S.and Jun, C.H., 2009. A simple and fast algorithm for K-medoids
      clustering.  Expert systems with applications, 36(2), pp.3336-3341.

    See also
    --------

    KMeans
        The KMeans algorithm minimizes the within-cluster sum-of-squares
        criterion. It scales well to large number of samples.

    Notes
    -----
    Since all pairwise distances are calculated and stored in memory for
    the duration of fit, the space complexity is O(n_samples ** 2).

    """

    def __init__(
        self,
        n_clusters=8,
        metric="euclidean",
        method="alternate",
        init="heuristic",
        max_iter=300,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.metric = metric
        self.method = method
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state

    def _check_nonnegative_int(self, value, desc, strict=True):
        """Validates if value is a valid integer > 0"""
        if strict:
            negative = (value is None) or (value <= 0)
        else:
            negative = (value is None) or (value < 0)
        if negative or not isinstance(value, (int, np.integer)):
            raise ValueError(
                "%s should be a nonnegative integer. "
                "%s was given" % (desc, value)
            )

    def _check_init_args(self):
        """Validates the input arguments."""

        # Check n_clusters and max_iter
        self._check_nonnegative_int(self.n_clusters, "n_clusters")
        self._check_nonnegative_int(self.max_iter, "max_iter", False)

        # Check init
        init_methods = ["random", "heuristic", "k-medoids++", "build"]
        if not (
            hasattr(self.init, "__array__")
            or (isinstance(self.init, str) and self.init in init_methods)
        ):
            raise ValueError(
                "init needs to be one of "
                + "the following: "
                + "%s" % (init_methods + ["array-like"])
            )

        # Check n_clusters
        if (
            hasattr(self.init, "__array__")
            and self.n_clusters != self.init.shape[0]
        ):
            warnings.warn(
                "n_clusters should be equal to size of array-like if init "
                "is array-like setting n_clusters to {}.".format(
                    self.init.shape[0]
                )
            )
            self.n_clusters = self.init.shape[0]

    def fit(self, X, len_cc_set=0):   # In X, the first len_cc_set points are cc_set, the number of X elements is n_sampling KMedoids KMedoids KMedoids KMedoids KMedoids
        """Fit K-Medoids to the provided data.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features) Dataset to cluster.
        len_cc_set : The medoids in X[len_cc_set] which does not need to be changed.

        Returns
        -------
        self
        """
        random_state_ = check_random_state(self.random_state)
        self._check_init_args()
        X = check_array(
            X, accept_sparse=["csr", "csc"], dtype=[np.float64, np.float32]
        )
        self.n_features_in_ = X.shape[1]
        if self.n_clusters > X.shape[0]:
            raise ValueError(
                "The number of medoids (%d) must be less "
                "than the number of samples %d."
                % (self.n_clusters, X.shape[0])
            )
        D = pairwise_distances(X, metric=self.metric)           # Here n_clusters is already c_set+cc_set, the number of X elements is n_sampling
        medoid_idxs = self._initialize_medoids(                 # The first len_cc_set in the returned medoid_idxs are range(len_cc_set), which are the indices of cc_set in X, these are all indices.
            D, self.n_clusters, random_state_, X, len_cc_set    # The first len_cc_set points are the indices of cc_set in X
        )                                                       
        # print("filterX_medoid_idxs: "+str(medoid_idxs))       # The first len_cc_set points are the indices of cc_set in X
        labels = None
        if self.method == "pam":
            # Compute the distance to the first and second closest points
            # among medoids.
            if self.n_clusters == 1 and self.max_iter > 0:
                # PAM SWAP step can only be used for n_clusters > 1
                warnings.warn(
                    "n_clusters should be larger than 2 if max_iter != 0 "
                    "setting max_iter to 0."
                )
                self.max_iter = 0
            elif self.max_iter > 0:
                Djs, Ejs = np.sort(D[medoid_idxs], axis=0)[[0, 1]]
        # Continue the algorithm as long as
        # the medoids keep changing and the maximum number
        # of iterations is not exceeded
        for self.n_iter_ in range(0, self.max_iter):
            old_medoid_idxs = np.copy(medoid_idxs)
            labels = np.argmin(D[medoid_idxs, :], axis=0)
            if self.method == "pam":
                not_medoid_idxs = np.delete(np.arange(len(D)), medoid_idxs)
                optimal_swap = _compute_optimal_swap(   # Although it will skip points in cc_set during updates, each updated point will consider points in cc_set when calculating distances to other points
                    D,
                    medoid_idxs.astype(np.intc),
                    not_medoid_idxs.astype(np.intc),
                    Djs,
                    Ejs,
                    self.n_clusters,        # Here n_clusters is already c_set+cc_set
                    len_cc_set              # Pass len_cc_set to Cython function
                )
                if optimal_swap is not None:
                    i, j, _ = optimal_swap
                    medoid_idxs[medoid_idxs == i] = j

                    # update Djs and Ejs with new medoids
                    Djs, Ejs = np.sort(D[medoid_idxs], axis=0)[[0, 1]]

            if np.all(old_medoid_idxs == medoid_idxs):
                break
            elif self.n_iter_ == self.max_iter - 1:
                warnings.warn(
                    "Maximum number of iteration reached before "
                    "convergence. Consider increasing max_iter to "
                    "improve the fit.",
                    ConvergenceWarning,
                )
        self.cluster_centers_ = X[medoid_idxs]      # Embedding of center points

        # Expose labels_ which are the assignments of
        # the training data to clusters
        self.labels_ = np.argmin(D[medoid_idxs, :], axis=0)
        self.medoid_indices_ = medoid_idxs
        self.inertia_ = _compute_inertia(self.transform(X))

        # Return self to enable method chaining
        return self

    def _update_medoid_idxs_in_place(self, D, labels, medoid_idxs):
        """In-place update of the medoid indices"""

        # Update the medoids for each cluster
        for k in range(self.n_clusters):
            # Extract the distance matrix between the data points
            # inside the cluster k
            cluster_k_idxs = np.where(labels == k)[0]

            if len(cluster_k_idxs) == 0:
                warnings.warn(
                    "Cluster {k} is empty! "
                    "self.labels_[self.medoid_indices_[{k}]] "
                    "may not be labeled with "
                    "its corresponding cluster ({k}).".format(k=k)
                )
                continue

            in_cluster_distances = D[
                cluster_k_idxs, cluster_k_idxs[:, np.newaxis]
            ]

            # Calculate all costs from each point to all others in the cluster
            in_cluster_all_costs = np.sum(in_cluster_distances, axis=1)

            min_cost_idx = np.argmin(in_cluster_all_costs)
            min_cost = in_cluster_all_costs[min_cost_idx]
            curr_cost = in_cluster_all_costs[
                np.argmax(cluster_k_idxs == medoid_idxs[k])
            ]

            # Adopt a new medoid if its distance is smaller then the current
            if min_cost < curr_cost:
                medoid_idxs[k] = cluster_k_idxs[min_cost_idx]

    def _compute_cost(self, D, medoid_idxs):
        """Compute the cose for a given configuration of the medoids"""
        return _compute_inertia(D[:, medoid_idxs])

    def transform(self, X):
        """Transforms X to cluster-distance space.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Data to transform.

        Returns
        -------
        X_new : {array-like, sparse matrix}, shape=(n_query, n_clusters)
            X transformed in the new space of distances to cluster centers.
        """
        X = check_array(
            X, accept_sparse=["csr", "csc"], dtype=[np.float64, np.float32]
        )

        if self.metric == "precomputed":
            check_is_fitted(self, "medoid_indices_")
            return X[:, self.medoid_indices_]
        else:
            check_is_fitted(self, "cluster_centers_")

            Y = self.cluster_centers_
            kwargs = {}
            if self.metric == "seuclidean":
                kwargs["V"] = np.var(np.vstack([X, Y]), axis=0, ddof=1)
            # DXY = pairwise_distances(X, Y=Y, metric=self.metric, **kwargs)
            DXY = compute_similarity_matrix(X, Y=Y)

            return DXY

    def predict(self, X):
        """Predict the closest cluster for each sample in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            New data to predict.

        Returns
        -------
        labels : array, shape = (n_query,)
            Index of the cluster each sample belongs to.
        """
        X = check_array(
            X, accept_sparse=["csr", "csc"], dtype=[np.float64, np.float32]
        )

        if self.metric == "precomputed":
            check_is_fitted(self, "medoid_indices_")
            return np.argmin(X[:, self.medoid_indices_], axis=1)
        else:
            check_is_fitted(self, "cluster_centers_")

            # Return data points to clusters based on which cluster assignment
            # yields the smallest distance
            kwargs = {}
            if self.metric == "seuclidean":
                kwargs["V"] = np.var(
                    np.vstack([X, self.cluster_centers_]), axis=0, ddof=1
                )
            # pd_argmin = pairwise_distances_argmin(
            #     X,
            #     Y=self.cluster_centers_,
            #     metric=self.metric,
            #     metric_kwargs=kwargs,
            # )
            pd_argmin = custom_pairwise_distances_argmin(X, Y=self.cluster_centers_)

            return pd_argmin

    def _initialize_medoids(self, D, n_clusters, random_state_, X=None, len_cc_set=0):  # This is to select initialization points from X_filter
        # Here n_clusters is already c_set+cc_set
        # In X, the first len_cc_set points are cc_set
        """Select initial mediods when beginning clustering."""
        # Here D is the distance matrix
        if hasattr(self.init, "__array__"):  # Pre assign cluster
            medoids = np.hstack(
                [np.where((X == c).all(axis=1)) for c in self.init]
            ).ravel()
        elif self.init == "random":  # Random initialization
            # Pick random k medoids as the initial ones.
            # medoids = random_state_.choice(len(D), n_clusters, replace=False)

            index_range = np.arange(len_cc_set, len(D))
            # Randomly select n_clusters-len_cc_set from the remaining elements
            medoids_others = random_state_.choice(index_range, n_clusters-len_cc_set, replace=False)
            medoids_cc_set = np.arange(len_cc_set)
            medoids = np.concatenate((medoids_cc_set, medoids_others))
            # print("medoids_cc_set: " + str(medoids_cc_set) + "medoids: " + str(medoids))
        elif self.init == "k-medoids++":
            medoids = self._kpp_init(D, n_clusters, random_state_)
        elif self.init == "heuristic":  # Initialization by heuristic
            # Pick K first data points that have the smallest sum distance
            # to every other point. These are the initial medoids.
            medoids = np.argpartition(np.sum(D, axis=1), n_clusters - 1)[
                :n_clusters
            ]
        elif self.init == "build":  # Build initialization
            # print("build_cc_set" + str(cc_set))
            if len_cc_set == 0:
                medoids = _build(D, n_clusters).astype(np.int64)
            else:
                # cc_set = [int(a) for a in cc_set]
                # cc_set = np.array(cc_set, dtype=np.intc) 
                medoids = _build_with_cc_set(D, n_clusters, len_cc_set).astype(np.int64)
                # print("initial check!!!!!!!" + str(cc_set) + str(medoids))
        else:
            raise ValueError(f"init value '{self.init}' not recognized")

        return medoids

    # Copied from sklearn.cluster.k_means_._k_init
    def _kpp_init(self, D, n_clusters, random_state_, n_local_trials=None):
        """Init n_clusters seeds with a method similar to k-means++

        Parameters
        -----------
        D : array, shape (n_samples, n_samples)
            The distance matrix we will use to select medoid indices.

        n_clusters : integer
            The number of seeds to choose

        random_state : RandomState
            The generator used to initialize the centers.

        n_local_trials : integer, optional
            The number of seeding trials for each center (except the first),
            of which the one reducing inertia the most is greedily chosen.
            Set to None to make the number of trials depend logarithmically
            on the number of seeds (2+log(k)); this is the default.

        Notes
        -----
        Selects initial cluster centers for k-medoid clustering in a smart way
        to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
        "k-means++: the advantages of careful seeding". ACM-SIAM symposium
        on Discrete algorithms. 2007

        Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
        which is the implementation used in the aforementioned paper.
        """
        n_samples, _ = D.shape

        centers = np.empty(n_clusters, dtype=int)

        # Set the number of local seeding trials if none is given
        if n_local_trials is None:
            # This is what Arthur/Vassilvitskii tried, but did not report
            # specific results for other than mentioning in the conclusion
            # that it helped.
            n_local_trials = 2 + int(np.log(n_clusters))

        center_id = random_state_.randint(n_samples)
        centers[0] = center_id

        # Initialize list of closest distances and calculate current potential
        closest_dist_sq = D[centers[0], :] ** 2
        current_pot = closest_dist_sq.sum()

        # pick the remaining n_clusters-1 points
        for cluster_index in range(1, n_clusters):
            rand_vals = (
                random_state_.random_sample(n_local_trials) * current_pot
            )
            candidate_ids = np.searchsorted(
                stable_cumsum(closest_dist_sq), rand_vals
            )

            # Compute distances to center candidates
            distance_to_candidates = D[candidate_ids, :] ** 2

            # Decide which candidate is the best
            best_candidate = None
            best_pot = None
            best_dist_sq = None
            for trial in range(n_local_trials):
                # Compute potential when including center candidate
                new_dist_sq = np.minimum(
                    closest_dist_sq, distance_to_candidates[trial]
                )
                new_pot = new_dist_sq.sum()

                # Store result if it is the best local trial so far
                if (best_candidate is None) or (new_pot < best_pot):
                    best_candidate = candidate_ids[trial]
                    best_pot = new_pot
                    best_dist_sq = new_dist_sq

            centers[cluster_index] = best_candidate
            current_pot = best_pot
            closest_dist_sq = best_dist_sq

        return centers


class CLARA(BaseEstimator, ClusterMixin, TransformerMixin):
    """CLARA clustering.

    Read more in the :ref:`User Guide <CLARA>`.
    CLARA (Clustering for Large Applications) extends k-medoids approach for a
    large number of objects. This algorithm use a sampling approach.

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of medoids to
        generate.

    metric : string, or callable, optional, default: 'euclidean'
        What distance metric to use. See :func:metrics.pairwise_distances

    max_iter : int, optional, default : 300
        Specify the maximum number of iterations when fitting PAM. It can be zero
        in which case only the initialization is computed.

    n_sampling : int or None, optional, default : None
        Size of the sampled dataset at each iteration. sampling-size a trade-off
        between complexity and efficiency. If None, then sampling-size is set
        to min(sample_size, 40 + 2 * self.n_clusters) as suggested by the authors of the
        algorithm. must be smaller than sample_size.

    n_sampling_iter : int, optional, default : 5
        Number of different samples that have to be done, or number of iterations.

    random_state : int, RandomState instance or None, optional
        Specify random state for the random number generator. Used to
        initialise medoids when init='random'.

    Attributes
    ----------
    cluster_centers_ : array, shape = (n_clusters, n_features)
            or None if metric == 'precomputed'
        Cluster centers, i.e. medoids (elements from the original dataset)

    medoid_indices_ : array, shape = (n_clusters,)
        The indices of the medoid rows in X

    labels_ : array, shape = (n_samples,)
        Labels of each point

    inertia_ : float
        Sum of distances of samples to their closest cluster center.

    Examples
    --------
    >>> from sklearn_extra.cluster import CLARA
    >>> import numpy as np
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(centers=[[0,0],[1,1]], n_features=2,random_state=0)
    >>> clara = CLARA(n_clusters=2, random_state=0).fit(X)
    >>> clara.predict([[0,0], [4,4]])
    array([0, 1])
    >>> clara.inertia_
    122.44919397611667

    References
    ----------
        Kaufman, L. and Rousseeuw, P.J. (2008). Partitioning Around Medoids (Program PAM).
        In Finding Groups in Data (eds L. Kaufman and P.J. Rousseeuw).
        doi:10.1002/9780470316801.ch2

    See also
    --------

    KMedoids
        CLARA is a variant of KMedoids that use sub-sampling scheme as such if the
        dataset is sufficiently small, KMedoids is preferable.

    Notes
    -----
    Contrary to KMedoids, CLARA is linear in N the sample size for both the spacial
    and time complexity. On the other hand, it scales quadratically with n_sampling.

    """

    def __init__(
        self,
        n_clusters=8,
        metric="euclidean",
        init="build",
        max_iter=300,
        n_sampling=None,
        n_sampling_iter=5,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.metric = metric
        self.init = init
        self.max_iter = max_iter
        self.n_sampling = n_sampling
        self.n_sampling_iter = n_sampling_iter
        self.random_state = random_state

    def fit(self, X, cc_set=[], y=None):
        """Fit CLARA to the provided data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features), \
                or (n_n_sampling_iter, n_n_sampling_iter) if metric == 'precomputed'
            Dataset to cluster.

        y : Ignored

        Returns
        -------
        self
        """
        X = check_array(X, dtype=[np.float64, np.float32])
        self.n_features_in_ = X.shape[1]

        n = len(X)  # n is the number of elements

        random_state_ = check_random_state(self.random_state)

        if self.n_sampling is None:
            n_sampling = max(
                min(n, 40 + 2 * self.n_clusters), self.n_clusters + 1
            )
        else:
            n_sampling = self.n_sampling

        # Check n_sampling.

        if n < self.n_clusters:
            raise ValueError(
                "sample_size should be greater than self.n_clusters"
            )

        if self.n_clusters >= n_sampling:
            raise ValueError(
                "sampling size must be strictly greater than self.n_clusters"
            )
        
        # if len(cc_set) == 0:
        #     medoids_idxs = random_state_.choice(
        #         np.arange(n), size=self.n_clusters, replace=False,  # 保证在抽取中心点时删除cc_set中的点; 这里的self.n_clusters是不包含cc_set数量的;
        #     )
        # elif len(cc_set) != 0:
        #     medoids_idxs = random_state_.choice(
        #         np.delete(np.arange(n), cc_set), size=self.n_clusters, replace=False,  # 保证在抽取中心点时删除cc_set中的点; 这里的self.n_clusters是不包含cc_set数量的;
        #     )
        #     medoids_idxs = np.concatenate((cc_set, medoids_idxs)) # 第一次时, 把cc_set也算入初始的聚类中心点中, 这里的cc_set放在前n个, 方便之后取
        #     self.n_clusters += len(cc_set)  

        if len(cc_set) != 0:
            self.n_clusters += len(cc_set)

        best_score = np.inf
        for _ in range(self.n_sampling_iter):

            if len(cc_set) == 0:
                medoids_idxs = random_state_.choice(
                    np.arange(n), size=self.n_clusters, replace=False,  # Ensure to remove points from cc_set when extracting center points; self.n_clusters here does not include cc_set count
                )
            elif len(cc_set) != 0:
                medoids_idxs = random_state_.choice(
                    np.delete(np.arange(n), cc_set), size=self.n_clusters - len(cc_set), replace=False,  # Ensure to remove points from cc_set when extracting center points; self.n_clusters here does not include cc_set count
                )
                medoids_idxs = np.concatenate((cc_set, medoids_idxs)) # First time, include cc_set in initial clustering center points, cc_set is placed in the first n positions for convenience in later retrieval

            if n_sampling > n:
                sample_idxs = np.arange(n)
                print("n_sampling > n!!!!!!!!!!! n: "+str(n))
            else:
                sample_idxs = np.concatenate(   # Data points used for k-medoids this time, hstack, total of n_sampling points
                    (
                        medoids_idxs,           # These are the indices of initialized center points in the original parameter matrix, where the first len(cc_set) are the indices of cc_set in the original matrix X
                        random_state_.choice(
                            np.delete(np.arange(n), medoids_idxs),  # Remove the indices of this initialization center points, n is the number of elements in X
                            size=n_sampling - self.n_clusters,  # Extract data points used for k-medoids this time
                            replace=False,
                        ),
                    )
                )
                
            pam = KMedoids(
                n_clusters=self.n_clusters,     # Here n_clusters is already c_set+cc_set
                metric=self.metric,
                method="pam",
                init=self.init,
                max_iter=self.max_iter,
                random_state=random_state_,
            )
            pam.fit(X[sample_idxs], len(cc_set)) # When entering k-medoids, the points are already selected, which is X[sample_idxs], where the first len(cc_set) are cc_set, the first len(cc_set) of sample_idxs are the indices of cc_set in the original matrix X
            self.cluster_centers_ = pam.cluster_centers_
            self.inertia_ = _compute_inertia(self.transform(X))

            if pam.inertia_ < best_score:
                best_score = self.inertia_
                best_medoids_idxs = pam.medoid_indices_
                best_sample_idxs = sample_idxs

        self.medoid_indices_ = best_sample_idxs[best_medoids_idxs]
        # print("self.medoid_indices_: " + str(self.medoid_indices_))
        self.labels_ = np.argmin(self.transform(X), axis=1)
        self.n_iter_ = self.n_sampling_iter

        return self

    def transform(self, X):
        """Transforms X to cluster-distance space.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Data to transform.

        Returns
        -------
        X_new : {array-like, sparse matrix}, shape=(n_query, n_clusters)
            X transformed in the new space of distances to cluster centers.
        """
        X = check_array(
            X, accept_sparse=["csr", "csc"], dtype=[np.float64, np.float32]
        )

        if self.metric == "precomputed":
            check_is_fitted(self, "medoid_indices_")
            return X[:, self.medoid_indices_]
        else:
            check_is_fitted(self, "cluster_centers_")

            Y = self.cluster_centers_
            # return pairwise_distances(X, Y=Y, metric=self.metric)
            return compute_similarity_matrix(X, Y=Y)

    def predict(self, X):
        """Predict the closest cluster for each sample in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            New data to predict.

        Returns
        -------
        labels : array, shape = (n_query,)
            Index of the cluster each sample belongs to.
        """
        X = check_array(
            X, accept_sparse=["csr", "csc"], dtype=[np.float64, np.float32]
        )

        if self.metric == "precomputed":
            check_is_fitted(self, "medoid_indices_")
            return np.argmin(X[:, self.medoid_indices_], axis=1)
        else:
            check_is_fitted(self, "cluster_centers_")

            # Return data points to clusters based on which cluster assignment
            # yields the smallest distance
            # return pairwise_distances_argmin(
            #     X, Y=self.cluster_centers_, metric=self.metric
            # )
            return custom_pairwise_distances_argmin(X, Y=self.cluster_centers_)
