import networkx as nx, pandas as pd, matplotlib.pyplot as plt


# TODO: Implement the degree centrality for a graph g
def degree_centrality(g):
    return [node_degree_tup[1] for node_degree_tup in g.degree()]


def solve_2_1():
    # Importing the graph (connected, unweighted, undirected social network)
    krackhardt_kite = nx.krackhardt_kite_graph()

    # TODO: Calculate and print the Kite's degree centrality
    kk_degree_centrality = [node_degree_tup[1] for node_degree_tup in krackhardt_kite.degree]
    print('Degree centrality:')
    print('\t\t\t'.join([f'node{i}: {dc}' for i, dc in enumerate(kk_degree_centrality)]))

    kk_closeness_centrality = nx.closeness_centrality(krackhardt_kite, normalized=False)
    print('Closeness centrality:')
    print('\t\t\t'.join([f'node{i}: {cc}' for i, cc in enumerate(kk_closeness_centrality)]))

def solve_2_2():
    global uni_net
    # TODO: Calculate and print the degree centrality using the degree_centrality function
    deg_cen = degree_centrality(uni_net)
    print(', '.join([f'node{i}: {dc}' for i, dc in enumerate(deg_cen)]))

    norm_const = 1.0 / (len(deg_cen) - 1.0)
    deg_cen_normalized = [deg * norm_const for deg in deg_cen]

    assert deg_cen_normalized == list(nx.degree_centrality(uni_net).values())
    uni_net.adj


def solve_2_3():
    # TODO: Calculate the betweenness centrality (using the pre-defined function is fine)
    btw_cen = nx.betweenness_centrality(uni_net, normalized=False)
    print('\nBetweeness Centrality:')
    print(btw_cen)


# Calculate the transition matrix element P_ij of a node i to a node j.
def pij(g, i, j):
    """
    calculate transition matrix element
    between node i and node j of graph g
    returns:
        1/"outdegree of node j",  if edge (j,i) exists
        0,                        otherwise
    """
    # TODO:
    if g.has_edge(j, i):
        deg_out_j = len(g.adj[j])
        # if deg_out_j == 0:
        #     return 1.0 / len(g.nodes)
        return 1.0 / deg_out_j
    return 0


# Renormalize after every step
def renormalize(pagerank_list):
    """
    input arbitrary float number list
    return a list where of all elements (sum(list)) equals 1.0
    """
    # TODO:
    return [float(pr) / sum(pagerank_list) for pr in pagerank_list]


def calcPageRank(g, d, numIter=3):
    """
    calculate the PageRank of a given graph g, with damping d,
    number of iterations numIter using jacobi power iteration
    return a list with pageranks.
    """
    # first initialize our pagerank centrality list c
    # with 1/N for each element
    N = len(g.nodes)
    c = [1.0 / N] * N  # TODO
    for iteration in range(numIter):

        c_previous = c.copy()

        for i in g.nodes:
            summe_j = 0.0

            for j in g.nodes:  # for neighbors of i
                # calculate the sum term
                summe_j += c_previous[j] * pij(g, i, j)  # TODO

            # calculate the centrality for the index i
            # using the complete formula
            c[i] = d * summe_j + (1.0 - d) / N  # TODO
        # renormalize pageranks after every iteration
        c = renormalize(c)

    return c


def solve_2_4():
    n = 20
    p = 0.07
    directed = True

    # TODO:
    g = nx.erdos_renyi_graph(n, p, directed=directed)

    # 2. Use the built-in function to calculate the PageRank.
    # Use these values for the following function
    ITERATIONS = 100
    DAMPING = 0.85

    # TODO:
    pagerank = nx.pagerank(g, alpha=DAMPING, max_iter=ITERATIONS)
    print('\nFirst 10 elements of PageRank:')
    print('\n'.join([f'node{n}: {pr}' for n, pr in pagerank.items()][:10]))

    # 5.
    my_pagerank_list = calcPageRank(g, DAMPING, ITERATIONS)
    pagerank_dict = nx.pagerank(g, alpha=DAMPING, max_iter=ITERATIONS)
    # You can use this to compare your results to the pre-defined function.
    # Please note that the results may vary by approx. 0.01. That is ok.
    # This is because NetworkX uses slightly different variation.
    for i in range(10):
        nx_pr = pagerank_dict[i]
        my_pr = my_pagerank_list[i]
        print(f"{nx_pr:.6f} -- {my_pr:.6f} (eps={abs(nx_pr - my_pr):.6f})")
    pass


if __name__ == '__main__':
    # solve_2_1()
    uni_net = nx.read_graphml('UniversityNetwork.graphml.xml')
    solve_2_2()

    solve_2_3()

    solve_2_4()
