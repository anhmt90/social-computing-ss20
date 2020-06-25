import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


def solve_a():
    indexNames = df_epchar[df_epchar.episode_id > HIGHEST_EPISODE].index
    df_epchar.drop(indexNames, inplace=True)


def solve_b():
    global df_merged
    df_merged = df_epchar.merge(df_nodes, how='inner', left_on='character_id', right_on='Id')
    df_merged.drop(['episode_id', 'character_id'], axis=1, inplace=True)

    # TODO: now we have unnecessary information, drop the duplicates
    df_merged.drop_duplicates(inplace=True, ignore_index=True)


def solve_c():
    global df_merged, df_merged2

    df_merged2 = df_merged.merge(df_edges, how='left', left_on='Id', right_on='Source')
    # Drop Type, as it is not that interesting
    df_merged2 = df_merged2.drop(['Type'], axis=1)


def solve_d():
    global df_merged2
    indexNames = df_merged2[df_merged2.Weight < 20].index
    df_merged2.drop(indexNames, inplace=True)


def solve_e():
    global df_merged2, graph

    # TODO:
    # Create a series for your character who is connected to homer 234 times
    # and add it to the dataframe
    cols = df_merged2.columns
    me_2_homer = pd.Series({cols[0]: 1337, cols[1]: 'Tuan Anh Ma', cols[2]: 1337, cols[3]: 1, cols[4]: 234})
    homer_2_me = pd.Series({cols[0]: 1, cols[1]: 'Homer Simpson', cols[2]: 1, cols[3]: 1337, cols[4]: 234})

    # TODO: append the list of series to the pandas data frame
    df_merged2 = df_merged2.append([me_2_homer, homer_2_me], ignore_index=True)
    df_merged2.sort_values(by='Id', inplace=True)
    # Create the graph from the dataframe
    graph = nx.from_pandas_edgelist(df_merged2, source="Id", target="Target", edge_attr=True)


def solve_f():
    global graph
    # Relabel the graph
    df_nodes_labels_dict = df_nodes.set_index('Id').to_dict()['charname']
    graph = nx.relabel_nodes(graph, df_nodes_labels_dict)

    # Det the edge color according to the weight
    edges, weights = zip(*nx.get_edge_attributes(graph, 'Weight').items())

    # Dtyle the graph
    options = {
        "font_size": 14,
        "font_color": '#552222',
        "node_color": '#22FF22',
        "width": 5.0,
        "edgelist": edges,
        "edge_color": weights,
        "edge_cmap": plt.cm.Blues
    }

    plt.figure(1, figsize=(40, 40))

    # TODO: plot the graph
    nx.draw(graph)


if __name__ == '__main__':
    # read the csv into pandas DataFrames
    df_edges = pd.read_csv("simpsons/edges.csv")
    df_nodes = pd.read_csv("simpsons/nodes.csv")
    df_epchar = pd.read_csv("simpsons/ep-char.csv")

    # 203 is the number of the last episode in season 9.
    HIGHEST_EPISODE = 203

    solve_a()

    df_merged = pd.DataFrame()
    solve_b()
    df_merged2 = pd.DataFrame()
    solve_c()
    solve_d()
    graph = nx.Graph()
    solve_e()
    solve_f()
