import scipy
import random
import pickle as pkl
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sc
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

from collections import Counter
from datetime import datetime, timedelta
from scipy.spatial import distance
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.metrics import precision_recall_fscore_support

# Set paramters for plotting
width = 7
height = width / 1.618
mpl.rcParams['axes.titlesize'] = 28
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 3
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['figure.figsize'] = (width, height)

# Reading in the pickle file
with open('Dataset.pkl', 'rb') as f:
    hourly_locs = pkl.load(f)
    feature_data = pkl.load(f)
    friends_set = pkl.load(f)
    not_friends_set = pkl.load(f)
    friends_survey = pkl.load(f)
    sbj_ids = pkl.load(f)


def solve_6_1():
    ## a)
    entropy_list = []
    # Loop over all UUIDs
    for uuid in range(107):
        entropy = None
        try:
            # Calculate frequency for each location
            places_f = Counter(
                [p for p_list in (hourly_locs[uuid][h] for h in range(24)) for p in p_list if not np.isnan(p)])
            # TODO: Calculate probability for each location
            total = sum(places_f.values())
            for place, occ in places_f.items():
                places_f[place] = occ / total

            # TODO: Compute entropy based on probabilities
            entropy = -sum(prob * np.log2(prob) for prob in places_f.values())
        # If UUID missing
        except:
            # TODO: Assign entropy -1
            entropy = -1
        entropy_list.append(entropy)
    print('Entropy for each UUID:', entropy_list)

    ## b)
    # TODO: Low entropy user
    uuid = 64
    # TODO: Get data for one month
    data = pd.DataFrame(hourly_locs[uuid]).iloc[:, :30].transpose().fillna(-1).astype('int32')

    # Create heatmap
    cmap = mpl.cm.YlOrRd
    fig, ax = plt.subplots(figsize=(1.7 * width, height))
    ax.set_title('Low Entropy User')
    sns.heatmap(data, cmap=cmap, ax=ax)
    plt.show()

    # TODO: High entropy user
    uuid = 102
    # TODO: Get data for one month
    data = pd.DataFrame(hourly_locs[uuid]).iloc[:, :30].transpose()

    # Create heatmap
    cmap = mpl.cm.YlOrRd
    fig, ax = plt.subplots(figsize=(1.7 * width, height))
    ax.set_title('High Entropy User')
    sns.heatmap(data, cmap=cmap, ax=ax)
    plt.show()

    print()


def _prior_6_2():
    ids_len = len(sbj_ids)
    friends_matrix = np.empty((ids_len, ids_len))

    # Create adjacency matrix for surveyed friendships
    for i, sid in enumerate(sbj_ids):
        friends_matrix[i] = [friends_survey[sid][sid2] for sid2 in sbj_ids]

    # Delete self loops
    np.fill_diagonal(friends_matrix, 0)

    # Create graph from adjacency matrix
    g = nx.from_numpy_matrix(friends_matrix)

    nx.draw(g)
    return friends_matrix, g


def solve_6_2():
    friends_matrix, g = _prior_6_2()

    gmm_accuracy = dict()
    predictions_dict = dict()
    ids_len = len(sbj_ids) - 1
    gmm_friends_matrix = np.empty((ids_len, ids_len))

    best = {
        'acc': 0.,
        'std': 0.,
        'n': 0
    }

    for n in range(1, 10):

        # Loop over all subject IDs
        for sid in list(feature_data.keys()):
            # Extract the features for current subject
            feature_table = feature_data[sid]
            x = feature_table.loc[:, :'callevent'].values

            # Fitting of the GMM to the features
            # TODO: Vary the number of components
            model = GaussianMixture(n_components=n, max_iter=500)
            model.fit(x)

            # Prediction of friendships between current subject and all others
            gmm_pred = model.predict(x)

            # Labels (0 or 1) are randomly assigned to 'friends' and 'not friends' but most common intuitively is the latter
            not_friend = Counter(gmm_pred).most_common()[0][0]

            # Create dict for predicted friendships
            predicted = pd.DataFrame({'subject': list(feature_table.index),
                                      'isfriendP': [int(label != not_friend) for label in gmm_pred]}).set_index('subject')
            predictions_dict[sid] = predicted

            # Evaluate which predictions match the actual friendship
            acc = feature_table.assign(isfriendP=predicted['isfriendP']).pipe(lambda df: df.isfriend == df.isfriendP)

            # TODO: Compute accuracy
            gmm_accuracy[sid] = acc.values.sum() / acc.size

            # Show accuracy for all users
        acc = pd.Series(gmm_accuracy).mean()
        std = pd.Series(gmm_accuracy).std()
        print('Overall accuracy: {:.4f} +/- {:.4f}'.format(acc, std))
    #     if round(best['acc'], 3) < round(acc, 3) or (round(best['acc'], 3) == round(acc, 3) and best['std'] > std):
    #         best['acc'] = acc
    #         best['std'] = std
    #         best['n'] = n
    #
    # print(best)

    # Create inferred network (adjacency matrix)
        for i, sid in enumerate(sbj_ids):
            try:
                gmm_friends_matrix[i - 1] = predictions_dict[sid].isfriendP.values
            except:
                gmm_friends_matrix[i - 1] = [0] * ids_len

        # Delete self loops
        np.fill_diagonal(gmm_friends_matrix, 0)

        # Create graph from adjacency matrix
        g = nx.from_numpy_matrix(gmm_friends_matrix)

        nx.draw(g)


####################################################### 6.3
# Calculate the class-conditonal density
def conditional_density(x, GMM):
    prob = 0

    # Sum over all mixture components
    for k in range(GMM.n_components):
        # Define the 6-dimensional normal distribution for one component using the GMM parameters
        func = multivariate_normal(mean=GMM.means_[k], cov=GMM.covariances_[k], allow_singular=True)

        # Evaluate the function at point x, multiply with component's weight
        prob += GMM.weights_[k] * func.pdf(x)

    return prob


# Classify the 6-dimensional point x by comparing both GMM evaluations
def GMM_classify(x, GMM_fr, GMM_not_fr, prior_fr, prior_not_fr):
    # TODO: Compute the conditional densities
    ccd_friend = conditional_density(x, GMM_fr)
    ccd_not_friend = conditional_density(x, GMM_not_fr)

    proba_friend = ccd_friend * prior_fr
    proba_not_friend = ccd_not_friend * prior_not_fr
    # TODO: Return the label with higher probability
    return 1 if proba_friend >= proba_not_friend else 0


def solve_6_3():
    # Split into training and test data by taking random samples the sets
    # TODO: Vary the number of feature vectors
    training_fr = list(random.sample(friends_set, 25))
    training_not_fr = list(random.sample(not_friends_set, 30))
    test = list(friends_set - set(training_fr)) + list(random.sample(set(not_friends_set - set(training_not_fr)), 15))

    # Compute the relative class frequencies (priors)
    num_pairs = len(training_fr) + len(training_not_fr)
    prior_fr = len(training_fr) / num_pairs
    prior_not_fr = len(training_not_fr) / num_pairs

    # Generate GMM for 'friends'
    x_fr = np.asarray([tup[0] for tup in training_fr])
    GMM_fr = GaussianMixture(n_components=5, max_iter=500).fit(x_fr)

    # Generate GMM for 'not friends'
    x_not_fr = np.asarray([tup[0] for tup in training_not_fr])
    GMM_not_fr = GaussianMixture(n_components=5, max_iter=500).fit(x_not_fr)

    estimates = []
    ground_truth = []

    for x in test:
        # Classify the pair x as friends (1) or no friends (0)
        label = GMM_classify(x[0], GMM_fr, GMM_not_fr, prior_fr, prior_not_fr)

        estimates.append(label)
        ground_truth.append(x[1])

    # Evaluating the classifier
    print('Prediction: ', estimates)
    print('Ground truth: ', ground_truth)

    performance = precision_recall_fscore_support(y_true=ground_truth, y_pred=estimates)
    print('\nPrecision:', performance[0])
    print('Recall:', performance[1])
    print('F1 score:', performance[2])


if __name__ == '__main__':
    # solve_6_1()
    # solve_6_2()
    solve_6_3()
