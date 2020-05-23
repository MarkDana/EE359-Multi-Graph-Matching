import numpy as np

LAMBDA = 0.3


def cal_affinity_score(X, K):
    """
    :param X: matching results, (num_graph, num_graph, num_node, num_node)
    :param K: affinity matrix, (num_graph, num_graph, num_node^2, num_node^2)
    :return: normalized affinity score, (num_graph, num_graph)
    """
    n, _, m, _ = X.shape
    vx = np.reshape(X.transpose((0, 1, 3, 2)), newshape=(n, n, -1, 1))
    vxT = vx.transpose((0, 1, 3, 2))
    affinity_score = np.matmul(np.matmul(vxT, K), vx)  # in shape (n, n, 1, 1)
    normalized_affinity_score = affinity_score.reshape(n, n) / np.max(affinity_score)
    return normalized_affinity_score


def cal_pairwise_consistency(X):
    """
    :param X: matching results, (num_graph, num_graph, num_node, num_node)
    :return: pairwise_consistency: (num_graph, num_graph)
    """
    n, _, m, _ = X.shape
    X_t = X.transpose((1, 0, 2, 3))
    # matmul:
    pairwise_consistency = 1 - np.abs(X[:, :, None] - np.matmul(X[:, None],X_t[None, ...])).sum((2, 3, 4)) / (2 * n * m)
    # point-wise:
    # pairwise_consistency = 1 - np.abs(X[:, :, None] - X_t[None, ...] * X[:, None]).sum((2, 3, 4)) / (2 * n * m)
    return pairwise_consistency


def mgm_floyd(X, K, num_graph, num_node):
    """
    :param X: matching results, (num_graph, num_graph, num_node, num_node)
    :param K: affinity matrix, (num_graph, num_graph, num_node^2, num_node^2)
    :param num_graph: number of graph, int
    :param num_node: number of node, int
    :return: matching results, (num_graph, num_graph, num_node, num_node)
    """
    for k in range(num_graph):
        pairwise_consistency = cal_pairwise_consistency(X)
        Xopt = np.matmul(X[:, k][:, None], X[k, :][None, ...])
        Sorg = (1 - LAMBDA) * cal_affinity_score(X, K) + LAMBDA * np.sqrt(pairwise_consistency)  # sqrt for pc
        Sopt = (1 - LAMBDA) * cal_affinity_score(Xopt, K) + LAMBDA * np.sqrt(
            np.matmul(pairwise_consistency[:, k][:, None], pairwise_consistency[k, :][None, ...]))
        update = (Sopt > Sorg)[:, :, None, None]
        X = update * Xopt + (1 - update) * X

    return X


if __name__ == '__main__':
    X = np.random.randint(0, 10, (2, 2, 3, 3))
    cal_pairwise_consistency(X)
