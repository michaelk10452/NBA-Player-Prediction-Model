from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguegamefinder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import cross_val_score

import pandas as pd

# Function to fetch player ID based on player name
def get_player_id(player_name):
    player_dict = players.get_players()
    player = [p for p in player_dict if p['full_name'].lower() == player_name.lower()]
    print(player)
    return player[0]['id'] if player else None

# Function to fetch opponent ID based on team abbreviation
def get_team_id(team_abbr):
    team_dict = teams.get_teams()
    team = [t for t in team_dict if t['abbreviation'].lower() == team_abbr.lower()]
    return team[0]['id'] if team else None

# Function to fetch games for the last 3 seasons
def get_last_3_seasons(player_id):
    seasons = ['2021-22', '2022-23', '2023-24']  # Last 3 seasons
    all_seasons_data = []
    for season in seasons:
        season_data = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
        all_seasons_data.append(season_data)
    return pd.concat(all_seasons_data, ignore_index=True)

# Function to fetch the last 5 games
def get_last_5_games(player_id):
    gamelog = playergamelog.PlayerGameLog(player_id=player_id, season='2024-25').get_data_frames()[0]
    return gamelog.head(5)

# Function to fetch last 3 matchups with opponent
def get_last_3_matchups(player_id, opponent_abbr):
    gamefinder = leaguegamefinder.LeagueGameFinder(player_or_team_abbreviation='P', player_id_nullable=player_id)
    games = gamefinder.get_data_frames()[0]
    matchups = games[games['MATCHUP'].str.contains(opponent_abbr)]
    return matchups.head(3)

# Function to calculate average points from a set of games
def calculate_average_points(games):
    return games['PTS'].mean()

# Print the last 5 games and the last 3 matchups
def print_stats(games, title):
    print(f"\n{title}:")
    print(games[['GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST', 'STL', 'BLK']])

# User input function
def user_input():
    player_name = input("Enter player name: ")
    points = int(input("Enter target points: "))
    opponent = input("Enter opponent team (3-letter abbreviation): ").upper()
    home_away = input("Is it a home game or away? (home/away): ")
    return player_name, points, opponent, home_away

# Prediction function using Logistic Regression with Cross-Validation
def predict_score_with_cv(player_id, opponent_abbr, target_points, home_away):
    # Fetch data for the past 3 seasons
    all_seasons_games = get_last_3_seasons(player_id)
    
    # Fetch the last 5 games and last 3 matchups
    last_5_games = get_last_5_games(player_id)
    last_3_matchups = get_last_3_matchups(player_id, opponent_abbr)

    # Combine the datasets: past 3 seasons and recent games
    recent_games = pd.concat([last_5_games, last_3_matchups]).drop_duplicates()
    all_games = pd.concat([all_seasons_games, recent_games]).drop_duplicates()

    # Features: Points, Rebounds, Assists, Steals, Blocks, Home/Away (encoded as 1 or 0)
    X = all_games[['PTS', 'REB', 'AST', 'STL', 'BLK']].values
    home_away_val = 1 if home_away == 'home' else 0
    X = np.hstack((X, np.full((X.shape[0], 1), home_away_val)))

    # Target: Whether points exceed target_points
    y = (all_games['PTS'] > target_points).astype(int)

    # Check if y contains both classes (0 and 1)
    if len(np.unique(y)) < 2:
        if np.all(y == 0):
            print(f"All data points are below the target of {target_points} points.")
            return 0, None  # Predict "below" with no model accuracy
        else:
            print(f"All data points are above the target of {target_points} points.")
            return 1, None  # Predict "above" with no model accuracy

    # Train Logistic Regression with cross-validation
    model = LogisticRegression()
    scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
    model.fit(X, y)

    # Model accuracy from cross-validation
    accuracy = scores.mean()

    # Prediction (with probabilities)
    prob = model.predict_proba([[target_points, 0, 0, 0, 0, home_away_val]])[0][1]  # Probability of scoring above target
    prediction = model.predict([[target_points, 0, 0, 0, 0, home_away_val]])

    return prediction[0], accuracy, prob

# Main program
if __name__ == "__main__":
    player_name, points, opponent, home_away = user_input()

    # Get player and opponent ID
    player_id = get_player_id(player_name)
    opponent_id = get_team_id(opponent)

    if player_id is None or opponent_id is None:
        print("Invalid player or team. Please check the input.")
    else:
        # Get last 5 games and last 3 matchups
        last_5_games = get_last_5_games(player_id)
        last_3_matchups = get_last_3_matchups(player_id, opponent)

        # Calculate averages
        avg_last_5_games = calculate_average_points(last_5_games)
        avg_last_3_matchups = calculate_average_points(last_3_matchups)

        # Print stats
        print(f"\nAverage points from the last 5 games: {avg_last_5_games:.2f}")
        print_stats(last_5_games, "Last 5 Games")
        print_stats(last_3_matchups, "Last 3 Matchups with the Opponent")
        print(f"\nAverage points from the last 3 matchups: {avg_last_3_matchups:.2f}")

        # Prediction with cross-validation
        prediction, accuracy, prob = predict_score_with_cv(player_id, opponent, points, home_away)

        # Check if prediction is None (not enough data to train)
        if prediction is not None:
            result = "above" if prediction == 1 else "below"
            print(f"\nThe model predicts {player_name} will score {result} {points} points.")
            print(f"Cross-Validated Model Accuracy: {accuracy * 100:.2f}%")
            print(f"Model's Confidence in scoring above {points} points: {prob * 100:.2f}%")
        else:
            print("Prediction could not be made due to insufficient data.")