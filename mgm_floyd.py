import numpy as np

LAMBDA = 0.65
X0 = 1


def cal_affinity_score(X, K):
    """
    :param X: matching results, (num_graph, num_graph, num_node, num_node)
    :param K: affinity matrix, (num_graph, num_graph, num_node^2, num_node^2)
    :return: normalized affinity score, (num_graph, num_graph)
    """
    n, _, m, _ = X.shape
    # UPDATE 0604: this is column vectorize
    XT = X.transpose((0, 1, 3, 2))
    vx = np.reshape(XT, newshape=(n, n, -1, 1))
    vxT = vx.transpose((0, 1, 3, 2))
    affinity_score = np.matmul(np.matmul(vxT, K), vx)  # in shape (n, n, 1, 1)
    normalized_affinity_score = affinity_score.reshape(n, n) / X0
    return normalized_affinity_score


def cal_pairwise_consistency(X):
    """
    :param X: matching results, (num_graph, num_graph, num_node, num_node)
    :return: pairwise_consistency: (num_graph, num_graph)
    """
    n, _, m, _ = X.shape
    X_t = X.transpose((1, 0, 2, 3))
    pairwise_consistency = 1 - np.abs(X[:, :, None] - np.matmul(X[:, None], X_t[None, ...])).sum((2, 3, 4)) / (
            2 * n * m)
    return pairwise_consistency


def mgm_floyd(X, K, num_graph, num_node):
    """
    :param X: matching results, (num_graph, num_graph, num_node, num_node)
    :param K: affinity matrix, (num_graph, num_graph, num_node^2, num_node^2)
    :param num_graph: number of graph, int
    :param num_node: number of node, int
    :return: matching results, (num_graph, num_graph, num_node, num_node)
    """
    global X0
    X0 = np.max(cal_affinity_score(X, K))

    for k in range(num_graph):
        Xopt = np.matmul(X[:, k, None], X[k, None, :])
        Sorg = cal_affinity_score(X, K)
        Sopt = cal_affinity_score(Xopt, K)

        update = (Sopt > Sorg)[:, :, None, None]
        update[np.diag_indices(num_graph)] = False
        X = update * Xopt + (1 - update) * X

    for k in range(num_graph):
        pairwise_consistency = cal_pairwise_consistency(X)
        Xopt = np.matmul(X[:, k, None], X[k, None, :])
        Sorg = (1 - LAMBDA) * cal_affinity_score(X, K) + LAMBDA * pairwise_consistency
        Sopt = (1 - LAMBDA) * cal_affinity_score(Xopt, K) + LAMBDA * np.sqrt(  # sqrt pc for approximate
            np.matmul(pairwise_consistency[:, k][:, None], pairwise_consistency[k, :][None, ...]))
        # Sopt = (1 - LAMBDA) * cal_affinity_score(Xopt, K) + LAMBDA * cal_pairwise_consistency(Xopt)
        update = (Sopt > Sorg)[:, :, None, None]
        update[np.diag_indices(num_graph)] = False
        X = update * Xopt + (1 - update) * X

    return X
