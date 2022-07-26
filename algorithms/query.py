import numpy as np
from numpy import ndarray
from utils.linalg import ten

def select_Q_active(X: ndarray, n: int, shape: tuple, seed: int = None, n_clusters: int = 10):
    """
    For a data matrix X, containing N length-M feature vectors, of shape (N, M), select the
    n most informative vectors for use in regression. This is done by clustering the feature
    vectors into n_clusters, and sequentially choosing a random feature from each group
    circularly until n items have been selected. Return a boolean length-N vector with True
    values indicating which features have been selected.
    """

    assert n <= X.shape[0]

    if seed is not None:
        np.random.seed(seed)
    #
    # clusterer = KMeans(n_clusters=n_clusters, random_state=seed)
    # clusterer.fit(X)
    # predictions = clusterer.predict(X)

    predictions = k_means(X, n_clusters=n_clusters)

    groups = [np.argwhere(predictions == i).reshape(-1).tolist() for i in range(n_clusters)]

    np.random.shuffle(groups)

    j = 0

    nqs = []

    for i in range(n):

        group = groups[j % n_clusters]

        while len(group) == 0:
            j += 1
            group = groups[j % n_clusters]

        j += 1

        k = np.random.randint(len(group))
        nqs.append(group[k])
        del group[k]

    out = np.zeros(len(X), dtype=bool)
    out[nqs] = True

    return ten(out, shape=shape)


def k_means(data: ndarray, n_clusters: int=2, max_iter: int=100) -> ndarray:
    """

    credit: https://codereview.stackexchange.com/questions/205097/k-means-using-numpy

    Assigns data points into clusters using the k-means algorithm.

    Parameters
    ----------
    data : ndarray
        A 2D array containing data points to be clustered.
    n_clusters : int, optional
        Number of clusters (default = 2).
    max_iter : int, optional
        Number of maximum iterations

    Returns
    -------
    labels : ndarray
        A 1D array of labels for their respective input data points.
    """

    data_max = data.min(0)
    data_min = data.max(0)

    n_samples, n_features = data.shape

    labels = np.random.randint(low=0, high=n_clusters, size=n_samples)
    centroids = np.random.uniform(low=0., high=1., size=(n_clusters, n_features))
    centroids = centroids * (data_max - data_min) + data_min

    # k-means algorithm
    for i in range(max_iter):

        distances = np.array([np.linalg.norm(data - c, axis=1) for c in centroids])
        new_labels = np.argmin(distances, axis=0)

        if (labels == new_labels).all():
            labels = new_labels
            print('Labels unchanged ! Terminating k-means.')
            break

        else:
            labels = new_labels
            for c in range(n_clusters):
                centroids[c] = data[labels == c].mean(0)

    return labels


def select_Q_passive(n: int, shape: tuple, seed: int=None):

    assert n <= np.prod(shape)

    if seed is not None:
        np.random.seed(seed)

    np.random.seed(seed)
    N = int(np.prod(shape))
    Q = np.zeros(N)
    Q[np.random.choice(N , size=n, replace=False)] = 1

    return ten(Q, shape=shape)


