import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import pickle

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import preprocessing
import numpy as np

from sklearn import svm
from sklearn.linear_model import LinearRegression
from statistics import mean
from sklearn import linear_model


def solve_4_1():
    global chat_data, chat_labels, predictions
    ### a)
    # TODO: Import "labels.xlsx" and split it into 2 arrays: chat and labels.
    df = pd.read_excel('labels.xlsx', header=None)
    chat_data = df[0][:450]
    chat_labels = df[1][:450]

    ### b)
    nltk.download('stopwords')
    chat_data = [str(item) for item in chat_data]
    vectorizer = TfidfVectorizer(max_features=2500, min_df=3, max_df=0.8, stop_words=stopwords.words('english'))
    chat_data = vectorizer.fit_transform(chat_data).toarray()

    # TODO: Create the random forest classifier, fit it and make a prediction on the test set.
    X_train, X_test, y_train, y_test = train_test_split(chat_data, chat_labels, test_size=0.2, random_state=0)
    # y_train --> ('neutral', 244) ('negative', 90) ('positive', 26)
    # y_test --> ('neutral', 56) ('negative', 30) ('positive', 4)

    rf = RandomForestClassifier(n_estimators=200, random_state=0)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    print('\nConfusion matrix:\n', confusion_matrix(y_test, predictions))
    print('\nClassification report:\n', classification_report(y_test, predictions))
    print('\nAccuracy score: ', accuracy_score(y_test, predictions))

    ### c)
    chatData = pd.read_csv("chat.csv")
    unlabeled = chatData.iloc[:, 1].values
    # TODO:

    # labeled_chats = {str(item) for item in df[0][:300]}
    unlabeled = [str(item) for item in unlabeled[450:]]
    unlabeled_tfidf = vectorizer.transform(unlabeled).toarray()
    predictions = rf.predict(unlabeled_tfidf)
    labels, counts = np.unique(predictions, return_counts=True)

    print('\nPrediction counts for unlabeled data: ', dict(zip(labels.tolist(), counts.tolist())))
    print(predictions)
    # preds --> [['negative', 'neutral', 'positive'], [3235, 42067, 1467]]


def solve_4_2():
    global chat_labels, predictions, sentiments
    # Convert the labels into values
    sentiments = []
    for i in chat_labels:
        if i == 'positive':
            sentiments.append(1)
        elif i == 'negative':
            sentiments.append(-1)
        else:
            sentiments.append(0)

    for i in predictions:
        if i == 'positive':
            sentiments.append(1)
        elif i == 'negative':
            sentiments.append(-1)
        elif i == 'neutral':
            sentiments.append(0)


def solve_4_2_a():
    global sentiments, full_chatdata, full_golddata, full_playerinfo, full_KDRatios, dataframe

    # 1. Read the csv files and group them by match id
    chatData = pd.read_csv("chat.csv")
    chatData = chatData.drop(['unit'], axis=1)
    chatData['label'] = sentiments  # We are assigning labels to the chat messages
    chatData = chatData.groupby('match_id')

    player_times = pd.read_csv("player_time.csv")
    player_times = player_times.groupby('match_id')

    match_info = pd.read_csv("match.csv")
    radiant_win = match_info['radiant_win']

    player_info = pd.read_csv("players.csv")
    player_info = player_info[['match_id', 'kills', 'deaths']]
    player_info = player_info.groupby('match_id')

    # 2. Create the dataframe
    dataframe = pd.DataFrame(columns=['chatData', 'goldData', 'KDratios', 'radiant_win'])

    # 3.
    full_chatdata = []

    for name, group in chatData:
        chat_data_line = []
        for index, row in group.iterrows():
            # TODO: Create a list of tuples called full_chatdata, each tuple has the following structure: label, team.
            # Hint 1: use the label column to determine the negativity/positivity of the message
            # Hint 2: use the 'slot' column to determine the team. 0 to 4 is for radiant, 5-9 is for dire.
            team = 'radiant' if 0 <= row['slot'] <= 4 else 'dire'
            chat_tuple = [row['label'], team]
            chat_data_line.append(chat_tuple)
        full_chatdata.append(chat_data_line)

    # 4. Create a list containing the gold advantage
    full_golddata = []

    for name, group in player_times:
        radiantAdv = []
        for index, row in group.iterrows():
            radiantAdv.append(
                (row['gold_t_0'] + row['gold_t_1'] + row['gold_t_2'] + row['gold_t_3'] + row['gold_t_4']) -
                (row['gold_t_128'] + row['gold_t_129'] + row['gold_t_130'] + row['gold_t_131'] + row['gold_t_132']))

        full_golddata.append(radiantAdv)

    # 5.
    full_playerinfo = []
    for name, group in player_info:
        playerinfo = []
        for index, row in group.iterrows():
            killsdeaths = [row['kills'], row['deaths']]
            playerinfo.append(killsdeaths)
        full_playerinfo.append(playerinfo)

    full_KDRatios = []

    for row in full_playerinfo:
        KDRatios = []
        ratiosRadiant = []
        ratiosDire = []
        assert len(row) == 10
        for i, player in enumerate(row):
            # TODO: Create a list called [...] kill-death ratios for each player
            # Hint: For each game the kd ratios should look like the following:
            # [[RadiantPlayer0KD, ... RadiantPlayer4KD],[DirePlayer0KD, ... DirePlayer4KD]]
            kills = player[0]
            deaths = player[1]
            ratio = (kills / deaths) if deaths != 0 else kills
            if i < 5:
                ratiosRadiant.append(ratio)
            else:
                ratiosDire.append(ratio)

        KDRatios.append(ratiosRadiant)
        KDRatios.append(ratiosDire)
        full_KDRatios.append(KDRatios)

    # We add the newly created columns to our dataframe
    dataframe['chatData'] = full_chatdata
    dataframe['goldData'] = full_golddata
    dataframe['radiant_win'] = radiant_win
    dataframe['KDratios'] = full_KDRatios

    dataframe.head(5)


def solve_4_2_b():
    global dataframe
    # 1. Average negativity
    radiantToxicity_full = []
    direToxicity_full = []

    for index, row in dataframe.iterrows():
        radiantToxicity = 0
        direToxicity = 0
        # These counters keep of track of the number of messages each team wrote:
        radiantcounter = 0
        direcounter = 0
        for tup in row['chatData']:
            # TODO: Calculate each team's toxicity by summing all labels of a match.
            # Hint: Don't forget to keep count of the number of messages written by each team.
            label, team = tup
            if team == 'radiant':
                radiantcounter += 1
                if label == -1:
                    radiantToxicity += label
            elif team == 'dire':
                direcounter += 1
                if label == -1:
                    direToxicity += label

        radiantToxicity_full.append(radiantToxicity / radiantcounter if radiantcounter != 0 else 0.0)
        direToxicity_full.append(direToxicity / direcounter if direcounter != 0 else 0.0)

    # 2. Average gold
    goldAverages = []
    goldEnd = []
    for index, row in dataframe.iterrows():
        # TODO: Compute the average gold advantage for each match, as well as the gold advantage at the end of the match.
        # Hint: The column goldData contains a list with gold advantage per minutes.
        goldAverages.append(sum(row['goldData']) / len(row['goldData']))
        goldEnd.append(row['goldData'][-1])

    # 3. Difference in negativity
    differences = []
    # TODO: Compute the difference in negativity between the 2 teams.
    differences = np.array(radiantToxicity_full) - np.array(direToxicity_full)
    differences = differences.tolist()

    # 4. K/D ratios
    worstRadiant = []
    worstDire = []
    for index, row in dataframe.iterrows():
        # TODO: Take the lowest K/D ratio from each team and create new columns for them.
        worstRadiant.append(min(row['KDratios'][0]))
        worstDire.append(min(row['KDratios'][1]))

    # We add the newly created columns to our dataframe
    dataframe['toxicityR'] = radiantToxicity_full
    dataframe['toxicityD'] = direToxicity_full
    dataframe['goldData'] = goldAverages
    dataframe['goldEnd'] = goldEnd
    dataframe['diff'] = differences
    dataframe['worstKDR'] = worstRadiant
    dataframe['worstKDD'] = worstDire

    dataframe.head(5)


def solve_4_2_c():
    global dataframe

    # TODO 1:
    for col_X in ['goldData', 'goldEnd']:
        print(f'Regression with single preditor X={col_X}')
        X = dataframe[col_X].to_numpy().reshape(-1, 1)  # TODO
        y = dataframe['radiant_win'].to_numpy()  # TODO

        # Splitting the data into training and testing data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        regr = LinearRegression()
        regr.fit(X_train, y_train)
        print(regr.score(X_test, y_test))

        y_pred = regr.predict(X_test)
        plt.scatter(X_test, y_test, color='b')
        plt.plot(X_test, y_pred, color='k')
        plt.show()


def solve_4_2_d():
    global dataframe

    # TODO 1:
    X = dataframe[dataframe.columns.difference(['chatData', 'radiant_win', 'KDratios'])].to_numpy()  # TODO
    Y = dataframe['radiant_win'].to_numpy()  # TODO

    # with sklearn
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    print(regr.score(X, Y))


if __name__ == '__main__':
    chat_data, chat_labels, predictions = None, None, None
    solve_4_1()

    sentiments = []
    solve_4_2()

    dataframe = pd.DataFrame()
    full_chatdata, full_golddata, full_playerinfo, full_KDRatios = [], [], [], []

    load_df = False
    if not load_df:
        solve_4_2_a()
        solve_4_2_b()
        with open('df.pkl', 'wb') as file:
            pickle.dump(dataframe, file)
    else:
        with open('df.pkl', 'rb') as file:
            dataframe = pickle.load(file)

    # solve_4_2_c()
    solve_4_2_d()
