import sys
sys.path.append('../')
from src.mgm_floyd import *

LAMBDA_SPFA = 0.3


def cal_pairwise_consistency_to_N(X):
    """
    :param X: matching results, (num_graph, num_graph, num_node, num_node)
    :return: pairwise_consistency: (num_graph,)
    """
    n, _, m, _ = X.shape
    fast_pc = 1 - np.abs(X[:, -1, None] - np.matmul(X, X[:, -1, None].transpose((1, 0, 2, 3)))).sum((1, 2, 3)) / (
            2 * n * m)
    return fast_pc


def cal_affinity_score_single(X, K):
    """
    :param X: matching results, (num_node, num_node)
    :param K: affinity matrix, (num_node^2, num_node^2)
    :return: normalized affinity score, (1,)
    """
    m, _ = X.shape
    # UPDATE 0604: np.reshape(X.T, newshape=(-1, 1)) this is column vectorize
    vx = np.reshape(X.T, newshape=(-1, 1))
    vxT = vx.transpose((1, 0))
    affinity_score = np.matmul(np.matmul(vxT, K), vx)
    normalized_affinity_score = affinity_score[0, 0] / X0
    return normalized_affinity_score


def cal_pairwise_consistency_single(X, i, j):
    """
    :param X: matching results, (num_graph, num_graph, num_node, num_node)
    :return: pairwise_consistency: (num_graph,)
    """
    n, _, m, _ = X.shape
    pc = 1 - np.abs(X[i, j] - np.matmul(X[i, ...], X[:, j])).sum((0, 1, 2)) / (
            2 * n * m)
    return pc


def mgm_spfa(K, X, num_graph, num_node):
    """
    :param K: affinity matrix, (num_graph, num_graph, num_node^2, num_node^2)
    :param X: matching results, X[:-1, :-1] is the matching results obtained by last iteration of MGM-SPFA,
              X[num_graph,:] and X[:,num_graph] is obtained via two-graph matching solver(RRWM), We suppose the last
              graph is the new coming graph. (num_graph, num_graph, num_node, num_node)
    :param num_graph: number of graph, int
    :param num_node: number of node, int
    :return: X, matching results, match graph_m to {graph_1, ... , graph_m-1)
    """

    for k in range(num_graph):
        Xopt = np.matmul(X[:, k, None], X[k, None, :])
        Sorg = cal_affinity_score(X, K)
        Sopt = cal_affinity_score(Xopt, K)
        update = (Sopt > Sorg)[:, :, None, None]
        update[np.diag_indices(num_graph)] = False
        X = update * Xopt + (1 - update) * X

    q = [i for i in range(num_graph - 1)]
    outnumber = 0
    while len(q) > 0:
        Gx = q[0]
        del q[0]
        outnumber += 1
        Xopt = np.matmul(X[:, Gx, None], X[Gx, None, :])  # X_opt[y,N]=X[y,x]·X[x,N]
        for y in range(num_graph - 1):
            if y == Gx:
                continue
            Sorg = (1 - LAMBDA_SPFA) * cal_affinity_score_single(X[y, -1], K[y, -1]) + \
                   LAMBDA_SPFA * cal_pairwise_consistency_single(X, y, -1)
            Sopt = (1 - LAMBDA_SPFA) * cal_affinity_score_single(Xopt[y, -1], K[y, -1]) + \
                   LAMBDA_SPFA * cal_pairwise_consistency_single(Xopt, y, -1)
            if Sorg < Sopt:
                X[y, -1] = Xopt[y, -1]
                if y not in q:
                    q.append(y)
        if outnumber > num_graph ** 2:
            break

    pairwise_consistency = cal_pairwise_consistency(X)
    Xopt = np.matmul(X[:, num_graph - 1][:, None], X[num_graph - 1, :][None, ...])  # X_opt[x,y]=X[x,N]·X[N,y]
    Sorg = (1 - LAMBDA_SPFA) * cal_affinity_score(X, K) + LAMBDA_SPFA * pairwise_consistency  # sqrt for pc
    Sopt = (1 - LAMBDA_SPFA) * cal_affinity_score(Xopt, K) + LAMBDA_SPFA * np.sqrt(
        np.matmul(pairwise_consistency[:, num_graph - 1][:, None], pairwise_consistency[num_graph - 1, :][None, ...]))
    update = (Sopt > Sorg)[:, :, None, None]
    update[num_graph - 1] = False
    update[:, num_graph - 1] = False  # Gx, Gy in H\GN
    update[np.diag_indices(num_graph)] = False
    X = update * Xopt + (1 - update) * X

    return X
