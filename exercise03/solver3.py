from collections import defaultdict
import json
import pickle
import requests
import urllib
import pandas as pd
import json
from urllib.request import Request, urlopen
from pandas.io.json import json_normalize
from requests.exceptions import HTTPError


def save_dict_to_csv(_dict, filename):
    df = pd.DataFrame({key: pd.Series(value) for key, value in _dict.items()})
    df.to_csv(filename, encoding='utf-8', index=False, header=['uid'])


def load_csv_to_dict(csv_path):
    _dict = defaultdict(list)
    df = pd.read_csv(csv_path, delimiter=',', header=0)
    for i, row in df.iterrows():
        # _dict[row[]]
        pass


def solve_3_1():
    global friendids, users_gamedicts, id
    # Use this if you want to work with the default IDs
    # You can replace these values with your own ID and API key
    key = "CB35B8F8DCE9135DDAA3B0328FCE0103"
    id = "76561198329838242"
    # url = "http://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/?key=" + key + "&steamids=" + id
    # r = requests.get(url)
    # data = r.json()
    #
    # # Get friendslist
    # request = Request(
    #     "http://api.steampowered.com/ISteamUser/GetFriendList/v0001/?key=" + key + "&steamid=" + id + "&relationship=friend")
    # response = urlopen(request)
    # elevations = response.read()
    # data = json.loads(elevations)
    # friendslist = data['friendslist']
    # friends = friendslist['friends']
    #
    # friendids = []
    # tempIDs = []
    # for friend in friends:
    #     friendids.append(friend['steamid'])
    #
    # print(len(friendids))
    # # get friends of friends:
    # x = 0
    # while x < len(friendids):
    #     friendID = friendids[x]
    #     request = Request(
    #         "http://api.steampowered.com/ISteamUser/GetFriendList/v0001/?key=" + key + "&steamid=" + friendID + "&relationship=friend")
    #     try:
    #         response = urlopen(request)
    #     except urllib.error.HTTPError as e:
    #         print('401')
    #     elevations = response.read()
    #     try:
    #         data = json.loads(elevations)
    #     except json.JSONDecodeError:
    #         print('couldnt decode')
    #     friendslist = data['friendslist']
    #     friends = friendslist['friends']
    #
    #     friendidsNew = []
    #     for friend in friends:
    #         friendidsNew.append(friend['steamid'])
    #
    #     tempIDs += friendidsNew
    #     x += 1
    #
    # friendids += tempIDs
    # friendids = list(dict.fromkeys(friendids))
    # friendids = list(set(friendids))
    # print(len(friendids))

    ###################################################################################
    # Trim the list of IDs to reasonable values:
    if len(friendids) > 250:
        friendids = friendids[:250]
    print(len(friendids))

    with open('friendids.pkl', 'ab') as file:
        pickle.dump(friendids, file)

    users_gamedicts = {}  # The dictionary containing all information for every ID
    gamedict = {}  # A dict containing information for one player

    # Get owned games of friendslist:
    request = Request(
        "http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key=" + key + "&steamid=" + id + "&include_appinfo=1&format=json")

    # TODO:
    # Open the URL and read the json response and retrieve the games of your friends and their playtime
    # Save the games into a dictionary with key=name and values=playtime
    # Hint 1: You can obtain the games a user owns with data['response']['games']
    # Hint 2: You can retrieve their playtime with game['playtime_forever']
    response = urlopen(request)
    elevations = response.read()
    data = json.loads(elevations)
    gamedict = {game['name']: game['playtime_forever'] for game in data['response']['games'] if
                game['playtime_forever'] != 0}

    # Add the dictionary to the users_gamedict
    users_gamedicts[int(id)] = gamedict

    # Do the same for the friends of your friends
    for friendID in friendids:
        # TODO
        request = Request(
            "http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key=" + key + "&steamid=" + friendID + "&include_appinfo=1&format=json")
        try:
            response = urlopen(request)
        except urllib.error.HTTPError as e:
            print(f'401: {e}')

        elevations = response.read()
        try:
            data = json.loads(elevations)
        except json.JSONDecodeError:
            print('couldnt decode')

        if data['response'].get('games'):
            gamedict = {game['name']: game['playtime_forever'] for game in data['response']['games'] if
                        game['playtime_forever'] != 0}
            if len(gamedict) > 0:
                users_gamedicts[int(friendID)] = gamedict

    with open('users_gamedicts.pkl', 'ab') as file:
        pickle.dump(users_gamedicts, file)


def load():
    global friendids, users_gamedicts

    with open('friendids.pkl', 'rb') as file:
        friendids = pickle.load(file)

    with open('users_gamedicts.pkl', 'rb') as file:
        users_gamedicts = pickle.load(file)
        users_gamedicts = {uid: games for uid, games in users_gamedicts.items() if len(games) > 0}


def solve_3_2():
    global users_gamedicts, gamesofallusers

    gamesofallusers = [list(gamedict.keys()) for fid, gamedict in users_gamedicts.items()]
    # TODO: Convert the gamedict to a list of lists

    # Remove common Steam entries that are not games:
    for game in gamesofallusers:
        if 'Dota 2 Test' in game:
            game.remove('Dota 2 Test')
        if 'True Sight' in game:
            game.remove('True Sight')
        if 'True Sight: Episode 1' in game:
            game.remove('True Sight: Episode 1')
        if 'True Sight: Episode 2' in game:
            game.remove('True Sight: Episode 2')
        if 'True Sight: Episode 3' in game:
            game.remove('True Sight: Episode 3')
        if 'True Sight: The Kiev Major Grand Finals' in game:
            game.remove('True Sight: The Kiev Major Grand Finals')
        if 'True Sight: The International 2017' in game:
            game.remove('True Sight: The International 2017')
        if 'True Sight: The International 2018 Finals' in game:
            game.remove('True Sight: The International 2018 Finals')

    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori

    te = TransactionEncoder()
    # TODO: Tinker around with the values
    te_ary = te.fit(gamesofallusers).transform(gamesofallusers)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
    frequent_itemsets.sort_values(by='support', ascending=False, ignore_index=True, inplace=True)

    import numpy as np
    from mlxtend.frequent_patterns import association_rules

    thresholds = np.arange(0.3, 1.0001, 0.01).tolist()
    for f in thresholds:
        f = round(f, 2)
        filtered_frequent_itemsets = frequent_itemsets[frequent_itemsets.support >= f]
        if len(filtered_frequent_itemsets.index) > 0:
            for t in thresholds:
                t = round(t, 2)
                conf_rules = association_rules(filtered_frequent_itemsets, metric="confidence", min_threshold=t)
                lift_rules = association_rules(filtered_frequent_itemsets, metric="lift", min_threshold=t)

                print(
                    f'f={f}, t={t}: {len(filtered_frequent_itemsets.index)} | {len(conf_rules.index)} | {len(lift_rules.index)}')
        if len(filtered_frequent_itemsets.index) <= 5:
            break

    # TODO: Play around with the treshold value
    # conf_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.75)
    # lift_rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.75)
    pass


# Here we will calculate the similarity score between two friends based on their common games:
def calculate_similarity(user1ID, user2ID):
    """
    Task 3.3
    """
    # TODO:
    games1 = users_gamedicts[user1ID]
    games2 = users_gamedicts[user2ID]

    common_games = games1.keys() & games2.keys()

    if not common_games:
        return 0.0

    distance = 0.0
    for game in common_games:
        playtime1 = games1[game]
        playtime2 = games2[game]
        distance += abs(playtime1 - playtime2) / float(playtime2)

    return 1.0 / (1.0 + distance)


def solve_3_4():
    global users_gamedicts, gamesofallusers, allGames

    # List of all games that are owned by at least 1 person:
    for user in gamesofallusers:
        for game in user:
            allGames.append(game)

    # TODO : Create a list of games owned by at least 3 people
    games_ownercounts = defaultdict(int)
    for game in allGames:
        games_ownercounts[game] += 1
    allGames = [game for game, num_owner in games_ownercounts.items() if num_owner >= 3]

    print(predict_ratings())


# Find out which games you do not own out of all games because we are only interested in recommendations for games that we do not own
def difference(allGames, usersgames):
    # TODO:
    return list(set.difference(set(allGames), set(usersgames)))


# Predict ratings based on the formula above for each unowned game
def predict_ratings():
    global friendids, users_gamedicts, games_ownercounts
    # TODO:
    """
    Hint: Iterate over all unowned games and for each game calculate a rating based
    on your friends playtime and similarity score
    """
    games_users_mapping = defaultdict(list)
    for u_id, games in users_gamedicts.items():
        for game in games:
            games_users_mapping[game].append(u_id)

    pred_playtimes = {}

    unowned_games = difference(allGames, users_gamedicts[id])
    for game in unowned_games:
        friend_ids = games_users_mapping[game]
        weighted_playtime_sum = 0.0  # numerator
        w_sum = 0.0  # denominator
        for friend_id in friend_ids:
            w = calculate_similarity(id, friend_id)
            w_sum += w

            v_playtime = users_gamedicts[friend_id][game]
            weighted_playtime_sum += w * v_playtime

        pred_playtimes[game] = weighted_playtime_sum / w_sum if w_sum != 0.0 else 0.0

    return pred_playtimes


if __name__ == '__main__':
    friendids = []
    users_gamedicts = {}
    gamesofallusers = []
    allGames = []
    games_ownercounts = defaultdict(int)
    id = 76561198329838242
    # solve_3_1()
    load()

    # with open('steamGameData.json') as jsonfile:
    #     users_gamedicts = json.loads(jsonfile.read())

    solve_3_2()
    solve_3_4()
