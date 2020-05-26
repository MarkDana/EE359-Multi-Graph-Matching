from src.mgm_floyd import *
from src.mgm_spfa import *

CMIN = 5

LAMBDA_FAST = 0.3


def fast_spfa(K, X, num_graph, num_node):
    """
    :param K: affinity matrix, (num_graph, num_graph, num_node^2, num_node^2)
    :param X: matching results, X[:-1, :-1] is the matching results obtained by last iteration of MGM-SPFA,
              X[num_graph,:] and X[:,num_graph] is obtained via two-graph matching solver(RRWM), We suppose the last
              graph is the new coming graph. (num_graph, num_graph, num_node, num_node)
    :param num_graph: number of graph, int
    :param num_node: number of node, int
    :return: X, matching results, match graph_m to {graph_1, ... , graph_m-1)
    """
    M = max(1, num_graph // CMIN)

    for k in range(num_graph):
        Xopt = np.matmul(X[:, k, None], X[k, None, :])
        Sorg = cal_affinity_score(X, K)
        Sopt = cal_affinity_score(Xopt, K)

        update = (Sopt > Sorg)[:, :, None, None]
        update[np.diag_indices(num_graph)] = False
        X = update * Xopt + (1 - update) * X

    for ci in range(M):
        if ci < M - 1:
            coord = [i for i in range(ci * CMIN, (ci + 1) * CMIN)]
            if num_graph - 1 not in coord:
                coord.append(num_graph - 1)
        else:
            coord = [i for i in range(ci * CMIN, num_graph)]
        Xc = X[coord, ...][:, coord]
        Kc = K[coord, ...][:, coord]

        q = [i for i in range(len(coord) - 1)]
        outnumber = 0
        while len(q) > 0:
            Gx = q[0]
            del q[0]
            outnumber += 1
            Xopt = np.matmul(Xc[:, Gx, None], Xc[Gx, None, :])  # X_opt[y,N]=X[y,x]·X[x,N]
            for y in range(len(coord) - 1):
                if y == Gx:
                    continue
                Sorg = (1 - LAMBDA_FAST) * cal_affinity_score_single(Xc[y, -1], Kc[y, -1]) + \
                       LAMBDA_FAST * cal_pairwise_consistency_single(Xc, y, -1)
                Sopt = (1 - LAMBDA_FAST) * cal_affinity_score_single(Xopt[y, -1], Kc[y, -1]) + \
                       LAMBDA_FAST * cal_pairwise_consistency_single(Xopt, y, -1)
                if Sorg < Sopt:
                    Xc[y, -1] = Xopt[y, -1]
                    if y not in q:
                        q.append(y)
            if outnumber > CMIN ** 2:
                break

    pairwise_consistency = cal_pairwise_consistency(X)
    Xopt = np.matmul(X[:, num_graph - 1][:, None], X[num_graph - 1, :][None, ...])  # X_opt[x,y]=X[x,N]·X[N,y]
    Sorg = (1 - LAMBDA_FAST) * cal_affinity_score(X, K) + LAMBDA_FAST * pairwise_consistency  # sqrt for pc
    Sopt = (1 - LAMBDA_SPFA) * cal_affinity_score(Xopt, K) + LAMBDA_SPFA * np.sqrt(
        np.matmul(pairwise_consistency[:, num_graph - 1][:, None], pairwise_consistency[num_graph - 1, :][None, ...]))
    update = (Sopt > Sorg)[:, :, None, None]
    update[num_graph - 1] = False
    update[:, num_graph - 1] = False  # Gx, Gy in H\GN
    update[np.diag_indices(num_graph)] = False
    X = update * Xopt + (1 - update) * X

    return X
