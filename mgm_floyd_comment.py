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

    SLOW_pointwise = np.zeros((n, n))
    SLOW_matmul = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cnt_pointwise = 0
            cnt_matmul = 0
            for k in range(n):
                cnt_pointwise += np.sum(np.abs(X[i, j] - X[i, k] * X[k, j]))
                cnt_matmul += np.sum(np.abs(X[i, j] - np.matmul(X[i, k], X[k, j])))
            SLOW_pointwise[i, j] = 1 - cnt_pointwise / (2 * n * m)
            SLOW_matmul[i, j] = 1 - cnt_matmul / (2 * n * m)

    # code 1
    X_t = X.transpose((1, 0, 2, 3))

    # print(X)
    # print(X_t[None, ...])
    # print(X[:, None])
    res = np.matmul( X[:, None],X_t[None, ...])
    pu = np.empty((n,n,n),dtype=object)

    for p in range(n):
        for q in range(n):
            for r in range(n):

                for i1 in range(n):
                    for j1 in range(n):
                        for i2 in range(n):
                            for j2 in range(n):
                                if (res[p,q,r]==np.matmul(X[i1,j1], X[i2,j2])).all():
                                    pu[p,q,r]='%d%dx%d%d' % (i1 + 1, j1 + 1, i2 + 1, j2 + 1)
                                    # print('%d%dx%d%d' % (i1 + 1, j1 + 1, i2 + 1, j2 + 1))


    print(pu)

    # for i1 in range(n):
    #     for j1 in range(n):
    #         for i2 in range(n):
    #             for j2 in range(n):
    #                 print('%d%dx%d%d'%(i1+1,j1+1,i2+1,j2+1))
    #                 print(np.matmul(X[i1,j1], X[i2,j2]))

    # matmul:
    # pairwise_consistency = 1 - np.abs(X[:, :, None] - np.matmul(X[:, None],X_t[None, ...])).sum((2, 3, 4)) / (2 * n * m)
    # point-wise:
    pairwise_consistency = 1 - np.abs(X[:, :, None] - X_t[None, ...] * X[:, None]).sum((2, 3, 4)) / (2 * n * m)

    # code 2
    my_pairwise_consistency = 1 - np.sum([np.abs(X - np.matmul(X[:, k], X[k, ...])).sum((2, 3)) for k in range(n)],
                                         (0,)) / (2 * n * m)
    #bug: X:(n,n,m,m), X[:, k]:(n,m,m), X[k, ...]:(n,m,m), and np.matmul(X[:, k], X[k, ...]) is still (n,m,m) !

    # code 3
    matmul_pairwise_consistency = 1 - np.sum([np.abs(X - np.matmul(X[:, None, k], X[k, None, ...])).sum((2, 3)) for k in range(n)],
                                         (0,)) / (2 * n * m)

    print((pairwise_consistency == SLOW_pointwise).all())
    print((pairwise_consistency == SLOW_matmul).all())
    print((my_pairwise_consistency == SLOW_matmul).all())
    print((matmul_pairwise_consistency == SLOW_matmul).all())

    return my_pairwise_consistency


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
