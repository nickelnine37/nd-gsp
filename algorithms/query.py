import numpy as np
from sklearn.cluster import KMeans
from numpy import ndarray


def select_Q_active(X: ndarray, n: int, seed: int = 0, n_clusters: int = 10):
    """
    For a data matrix X, containing N length-M feature vectors, of shape (N, M), select the
    n most informative vectors for use in regression. This is done by clustering the feature
    vectors into n_clusters, and sequentially choosing a random feature from each group
    circularly until n items have been selected. Return a boolean length-N vector with True
    values indicating which features have been selected.
    """

    np.random.seed(seed)

    clusterer = KMeans(n_clusters=n_clusters, random_state=seed)
    clusterer.fit(X)
    predictions = clusterer.predict(X)

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

    return out