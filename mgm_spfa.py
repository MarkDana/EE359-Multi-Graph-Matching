import sys
sys.path.append('../')
from src.mgm_floyd import *
import queue

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
    q = queue.Queue()
    outnumber = 0
    [q.put(i) for i in range(num_graph - 1)]
    pairwise_consistency = cal_pairwise_consistency(X)
    while not q.empty():
        Gx = q.get()
        outnumber += 1

        Xopt = np.matmul(X[:, Gx][:, None], X[Gx, :][None, ...])  # X_opt[y,N]=X[y,x]·X[x,N]
        Sorg = (1 - LAMBDA) * cal_affinity_score(X, K) + LAMBDA * np.sqrt(pairwise_consistency)  # sqrt for pc
        Sopt = (1 - LAMBDA) * cal_affinity_score(Xopt, K) + LAMBDA * np.sqrt(
            np.matmul(pairwise_consistency[:, Gx][:, None], pairwise_consistency[Gx, :][None, ...]))

        Sorg = Sorg[:, num_graph - 1, None, None]
        Sopt = Sopt[:, num_graph - 1, None, None] # only consider the new added one
        update = (Sopt > Sorg)
        update[Gx] = False # skip Gx the graph itself

        X[:, num_graph - 1] = update * Xopt[:, num_graph - 1] + (1 - update) * X[:, num_graph - 1]
        X[num_graph - 1, :] = update * Xopt[num_graph - 1, :].transpose((0, 2, 1)) + (1 - update) * X[num_graph - 1, :]

        # TODO: time test failed. Here i calculate all the pairwise X_opt[i,j], while only X_opt[i,N] is needed
        # TODO: however this may not downshift a lot, since it's parallelized computation
        # TODO: what inpacts most is to add Gy back into Q
        # TODO: if you comment the following one line, then accuracy and time test both passed
        # [q.put(y) for y in range(num_graph) if update[y]]  # add Gy into Q
        
        if outnumber % 2 == 0:
            pairwise_consistency = cal_pairwise_consistency(X)
        if outnumber > num_graph ** 2:
            break

    Xopt = np.matmul(X[:, num_graph - 1][:, None], X[num_graph - 1, :][None, ...])  # X_opt[x,y]=X[x,N]·X[N,y]
    Sorg = (1 - LAMBDA) * cal_affinity_score(X, K) + LAMBDA * np.sqrt(pairwise_consistency)  # sqrt for pc
    Sopt = (1 - LAMBDA) * cal_affinity_score(Xopt, K) + LAMBDA * np.sqrt(
        np.matmul(pairwise_consistency[:, num_graph - 1][:, None], pairwise_consistency[num_graph - 1, :][None, ...]))
    update = (Sopt > Sorg)[:, :, None, None]
    update[num_graph - 1] = False
    update[:, num_graph - 1] = False #Gx, Gy in H\GN
    X = update * Xopt + (1 - update) * X

    return X
