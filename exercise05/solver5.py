import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import seaborn as sns
from sklearn.manifold import TSNE


def solve_5_1():
    global df
    df = pd.read_csv('EVEPlayerStats.csv', header=0)
    df = df.drop(columns=['Unnamed: 0', 'characterID'])


def solve_5_2():
    global df, kmeans

    # TODO 1: Normalize all values by dividing by the max value for each column.
    df = df / df.max()

    # TODO 2: Convert the absolute numbers into ratios.
    # df['totalShipsLost'] = df[['combatShipsLost', 'exploShipsLost', 'miningShipsLost', 'otherShipsLost']].sum(axis=1)
    df['totalShipsLost'] = df.iloc[:, :4].sum(axis=1)
    df.iloc[:, :4] = df.iloc[:, :4].div(df['totalShipsLost'], axis=0)
    df = df.drop(columns='totalShipsLost')

    # TODO 3: Cluster the dataset with the k-Means algorithm.
    kmeans = KMeans(n_clusters=4, random_state=0).fit(df)
    # kmeans.cluster_centers_


# t-SNE Graph
def tsne(tempData):
    global kmeans

    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(tempData)

    new = tempData.copy()
    new['tsne-2d-one'] = X_2d[:, 0]
    new['tsne-2d-two'] = X_2d[:, 1]

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue=kmeans.labels_,
        palette=sns.color_palette("hls", 4),
        data=new,
        legend="full"
    )
    plt.show()


def solve_5_3():
    global df

    # a)
    sns.heatmap(df.iloc[:20])
    plt.show()
    # b)
    tsne(df)


if __name__ == '__main__':
    df = pd.DataFrame()
    kmeans = None

    solve_5_1()
    solve_5_2()
    solve_5_3()
