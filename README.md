# EE359-Multi-Graph-Matching
Python implementation of three algorithms in paper "Unifying Offline and Online Multi-graph Matching via Finding Shortest Paths on Supergraph". Homework of EE359, Prof. Junchi Yan.

<center>
#EE359 HW Report
###戴昊悦 李竞宇
###517030910{288,318}
-
</center>

### 1. Overview of the HW
In this short project we're required to implement Python code of the three algorithms proposed in [Jiang, Z., Wang, T., & Yan, J. (2020). Unifying Offline and Online Multi-graph Matching via Finding Shortest Paths on Supergraph. IEEE Transactions on Pattern Analysis and Machine Intelligence, 14(8), 1–1.](https://doi.org/10.1109/tpami.2020.2989928).

As follows we'll illustrate how we reproduce the algorithms, the result of pass test, and some of our observations.

### 2. Implementation

+ **Affinity Score**

In the graph matching synthesis, affinity score is designed to measure two-graph matching, usually written as a quadratic assignment programming (QAP) problem which is also called Lawler’s QAP:
$$
J ( \mathbf { X } ) = \min _ { \mathbf { X } \in \{ 0,1 \} ^ { n _ { 1 } \times n _ { 2 } }} \operatorname { vec } ( \mathbf { X } ) ^ { \top } \mathbf { K } \operatorname { vec } ( \mathbf { X } )  
$$
where $\mathbf { X }$ is a (partial) permutation matrix indicating the node correspondence, and $\mathbf { K } \in \mathbb{R}^{n1 n2 ×n1 n2}$ is the affinity matrix whose diagonal (off-diagonal) encodes the node-to-node affinity (edge-to-edge affinity) between two graphs. The symbol $\operatorname { vec } (·)$ here denotes the column-wise vectorization of the input matrix.

In our practice `X` is the matching result of multi graphs in shape `(num_graph, num_graph, num_node, num_node)`, where $\mathbf { X }$ in above formula is `X[i,j]`. Thus instead of calculating each pair of graphs, we can compute them in a bunch:

```python
def cal_affinity_score(X, K):
    """
    :param X: matching results, (num_graph, num_graph, num_node, num_node)
    :param K: affinity matrix, (num_graph, num_graph, num_node^2, num_node^2)
    :return: normalized affinity score, (num_graph, num_graph)
    """
    n, _, m, _ = X.shape
    # UPDATE: this is column vectorize
    XT = X.transpose((0, 1, 3, 2))
    vx = np.reshape(XT, newshape=(n, n, -1, 1))
    vxT = vx.transpose((0, 1, 3, 2))
    affinity_score = np.matmul(np.matmul(vxT, K), vx)  # in shape (n, n, 1, 1)
    normalized_affinity_score = affinity_score.reshape(n, n) / X0
    return normalized_affinity_score
```
Note that affinity score is normalized to range `(0,1]` to be consistent with pairwise consistency. We use the normalization factor `X0` to be the maximal affinity score of the raw input $X$, as proposed in CAO.

+ **Pairwise Consistency**

In the proposed unified approaches, given $\{G_k\}_{k-1}^{N}$ and matching configuration $\mathbb{X}$, for any pair $G_i$ and $G_j$, the pairwise consistency is defined as:
$$
C _ { p } \left( \mathbf { X } _ { i j } , \mathbb { X } \right) = 1 - \frac { \sum _ { k = 1 } ^ { N } \left\| \mathbf { X } _ { i j } - \mathbf { X } _ { i k } \mathbf { X } _ { k j } \right\| _ { F }  } {2 n N } \in ( 0,1 ]
$$
Though it's defined in a `for any` way and `k` is traversed as $\sum_{k=1}^{N}$, we don't need to write the code with three `for` loop, since it's mutually independent to compute each pair $G_i$ and $G_j$, as well as the summation of `k`.

Computation of $C _ { p } \left( \mathbf { X } _ { i j }\right)$ is related to $\mathbf { X } _ { i k } \mathbf { X } _ { k j } $ for $k$ from $1$ to $N$. This is similar to the form of matrix multiplication. However we want one step before: where $\mathbf { X } _ { i k } \mathbf { X } _ { k j } $ haven't been summarized so that $\left\| \mathbf { X } _ { i j } - \mathbf { X } _ { i k } \mathbf { X } _ { k j } \right\| _ { F }$ operation can be done. Thus we use `broadcasting` in `numpy` to align shapes with additional dimensions we add. Note that we need to swap the two axes with `transpose` to achieve it:

```python
def cal_pairwise_consistency(X):
    """
    :param X: matching results, (num_graph, num_graph, num_node, num_node)
    :return: pairwise_consistency: (num_graph, num_graph)
    """
    n, _, m, _ = X.shape
    X_t = X.transpose((1, 0, 2, 3)) # so that X_t[j,k] = X[k,j]
    pairwise_consistency = 1 - np.abs(X[:, :, None] - \
    	  np.matmul(X[:, None], X_t[None, ...])).sum((2, 3, 4)) / (2 * n * m)
    # X[:,None]*X_t[None,...] is X[i,k]*X[k,j] (matmul)
    return pairwise_consistency
```
We've had questions about whether to use pointwise or matrix multiplication here, which will be pointed out later.

+ **MGM-Floyd**

MGM-Floyd is used for offline multiple graph matching. It's able to find the optimal composition path more efficiently with fewer comparisons and thus being more competitive. Pseudocode provided in the paper:

<center><img src="imgs/floyd.png" width=50%/>
</center>

where $S \left( \mathbf { X } _ { i j } , \mathbb { X } \right) = \overbrace { ( 1 - \lambda ) J \left( \mathbf { X } _ { i j } \right) } ^ { \text{affinity score} } + \overbrace { \lambda C _ { p } \left( \mathbf { X } _ { i j } , \mathbb { X } \right) } ^ { \text{pairwise consistency} }$.  In practice we use the pc approximated version $S_{pc}^{\mathbb{X}} \left( \mathbf { X } _ { i j } , \mathbf { X } _ { jk } \right)=( 1 - \lambda ) J \left( \mathbf { X } _ { i j }\mathbf { X } _ { jk } \right) + \lambda \sqrt{C _ { p } \left( \mathbf { X } _ { i j } , \mathbb { X } \right)C _ { p } \left( \mathbf { X } _ { jk } , \mathbb { X } \right)}$. In this way we don't need to calculate pairwise consistency of the multiplied matrix again, but just multiply their original pairwise consistency value.

There are two rounds of updating $\mathbf{X}$, with each round traversing all graphs. In the first round $\lambda$ is set to $0$ for affinity based boosting. In the second round $\lambda=0.3$. Similar as acceleration of above, each pair of graphs are computated parallelly.

```python
def mgm_floyd(X, K, num_graph, num_node):
    """
    :param X: matching results, (num_graph, num_graph, num_node, num_node)
    :param K: affinity matrix, (num_graph, num_graph, num_node^2, num_node^2)
    :param num_graph: number of graph, int
    :param num_node: number of node, int
    :return: matching results, (num_graph, num_graph, num_node, num_node)
    """
    for k in range(num_graph):
        Xopt = np.matmul(X[:, None, k], X[None, k, :])
        Sorg = cal_affinity_score(X, K)
        Sopt = cal_affinity_score(Xopt, K)
        update = (Sopt > Sorg)[:, :, None, None]
        for i in range(num_graph):
          update[i, i] = False
        X = update * Xopt + (1 - update) * X

    for k in range(num_graph):
        pairwise_consistency = cal_pairwise_consistency(X)
        Xopt = np.matmul(X[:, None, k], X[None, k, :])
        Sorg = (1 - LAMBDA) * cal_affinity_score(X, K) + \
        	  LAMBDA * pairwise_consistency
        Sopt = (1 - LAMBDA) * cal_affinity_score(Xopt, K) + \
        	  LAMBDA * np.sqrt(\  # sqrt pc for approximate
            np.matmul(pairwise_consistency[:, k][:, None], \
            pairwise_consistency[k, :][None, ...]))
        update = (Sopt > Sorg)[:, :, None, None]
        update[np.diag_indices(num_graph)] = False
        X = update * Xopt + (1 - update) * X

    return X
```

We find that when using update in the matrix form would bring undesired update of self-assignment (i.e. $X_{ii}$), which should always be unit matrix. So we do not update these entries.

+ **MGM-SPFA**

MGM-SPFA is based on SPFA, a single-source shortest path algorithm. It helps solve online multiple graph matching, which aims at matching the arriving graph $G_N$ to $N − 1$ previous graphs which have already been matched. Two constraints added: force termination when number of updated nodes reaches $m^2$.

<center><img src="imgs/mgm-spfa.png" width=50%/>
</center>

```python
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

```

Here we use iteration instead of matrix operation to update the matches, because matrix operation has some redundancies: all the pairwise `X_opt[i,j]` are calculated, while only `X_opt[i,N]` is needed. Through experiments, we find that using iteration is more efficient than matrix operation in mgm-spfa. 

+ **FAST-SPFA**

Fast-SPFA is based on MGM-SPFA. But instead of doing MGM-SPFA on all of the graphs, Fast-SPFA randomly partition the graphs into several clusters and doing MGM-SPFA-like update on each clusters instead. In this part, all clusters updates match information by the newly arrived graph. Then in the second part, all graph updates their match to each other (regardless of clusters) given the newly arrived graph.

<center><img src="imgs/fast_spfa.png" width=50%/>
</center>

```python
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

```

-
### 2. Question and Observation
<center><img src="imgs/matmul0.png" width=30%/> <img src="imgs/matmul1.png" width=30%/>

<font color="grey">We are curious about whether $X_{ik}X_{kj}$ should be pointwise or matrix multiplication, for each of the cases</font>
</center>
Author Jiang Zetian has kindly answered our question:
><img src="imgs/whymatmul.png" width=50%/>

From the aspect of matching chain combination, it should be matmul here to maintain a permutation matrix. What we found interesting, however, is that if pointwise-mul is used in pairwise consistency computation rather than mat-mul, higher accuracy can be achieved (let alone speed). 

Another observation is that, when adding affinity boost step before both MGM-SPFA and Fast-SPFA, the accuracy would be increased dramatically while using little extra time.

We would like to keep the above two here and research more thoroughly in the next big project.


-
### 3. Results Screenshots

<center>
<table>
<thead>
  <tr>
    <th colspan="2">Offline Floyd</th>
    <th>Car</th>
    <th>Motorbike</th>
    <th>Face</th>
    <th>Winebottle</th>
    <th>Duck</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">Time Cost<br>(s)</td>
    <td>Required</td>
    <td>4.384</td>
    <td>4.227</td>
    <td>4.220</td>
    <td>4.339</td>
    <td>4.209</td>
  </tr>
  <tr>
    <td>Ours</td>
    <td>2.3236</td>
    <td>2.1036</td>
    <td>2.4006</td>
    <td>2.2967</td>
    <td>2.2574</td>
  </tr>
  <tr>
    <td rowspan="2">Accuracy<br>(%)</td>
    <td>Required</td>
    <td>60.46</td>
    <td>80.51</td>
    <td>91.08</td>
    <td>72.20</td>
    <td>57.69</td>
  </tr>
  <tr>
    <td>Ours</td>
    <td>82.16</td>
    <td>90.89</td>
    <td>95.77</td>
    <td>77.07</td>
    <td>75.20</td>
  </tr>
</tbody>
</table>

<table>
<thead>
  <tr>
    <th colspan="2">Online MGM-SPFA</th>
    <th>Car</th>
    <th>Motorbike</th>
    <th>Face</th>
    <th>Winebottle</th>
    <th>Duck</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">Time Cost<br>(s)</td>
    <td>Required</td>
    <td>2.190</td>
    <td>2.179</td>
    <td>2.023</td>
    <td>2.631</td>
    <td>2.135</td>
  </tr>
  <tr>
    <td>Ours</td>
    <td>1.021</td>
    <td>1.138</td>
    <td>1.046</td>
    <td>1.053</td>
    <td>1.158</td>
  </tr>
  <tr>
    <td rowspan="2">Accuracy<br>(%)</td>
    <td>Required</td>
    <td>63.32</td>
    <td>83.43</td>
    <td>91.41</td>
    <td>75.23</td>
    <td>59.05</td>
  </tr>
  <tr>
    <td>Ours</td>
    <td>80.65</td>
    <td>91.78</td>
    <td>96.46</td>
    <td>78.48</td>
    <td>77.89</td>
  </tr>
</tbody>
</table>

<table>
<thead>
  <tr>
    <th colspan="2">Online Fast-SPFA</th>
    <th>Car</th>
    <th>Motorbike</th>
    <th>Face</th>
    <th>Winebottle</th>
    <th>Duck</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">Time Cost<br>(s)</td>
    <td>Required</td>
    <td>0.7323</td>
    <td>0.7586</td>
    <td>0.8077</td>
    <td>0.7993</td>
    <td>0.7591</td>
  </tr>
  <tr>
    <td>Ours</td>
    <td>0.3661</td>
    <td>0.3376</td>
    <td>0.3507</td>
    <td>0.3663</td>
    <td>0.3285</td>
  </tr>
  <tr>
    <td rowspan="2">Accuracy<br>(%)</td>
    <td>Required</td>
    <td>61.87</td>
    <td>82.90</td>
    <td>91.45</td>
    <td>73.57</td>
    <td>57.82</td>
  </tr>
  <tr>
    <td>Ours</td>
    <td>80.51</td>
    <td>91.55</td>
    <td>96.46</td>
    <td>78.31</td>
    <td>77.81</td>
  </tr>
</tbody>
</table>

</center>

<center><img src="imgs/floyd_res.png" width=60%/>

<font color="grey">Offline test using MGM-floyd passed</font>
</center>

<center><img src="imgs/spfa1.png" width=18%/> <img src="imgs/spfa2.png" width=18%/> <img src="imgs/spfa3.png" width=18%/> <img src="imgs/spfa4.png" width=18%/> <img src="imgs/spfa5.png" width=18%/>

<font color="grey">Online test using MGM-SPFA passed</font>
</center>

<center><img src="imgs/fast1.png" width=18%/> <img src="imgs/fast2.png" width=18%/> <img src="imgs/fast3.png" width=18%/> <img src="imgs/fast4.png" width=18%/> <img src="imgs/fast5.png" width=18%/>

<font color="grey">Online test using Fast-SPFA passed</font>
</center>

P.S. Our experiments is run on Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz  1.99GHz, 16GB RAM.