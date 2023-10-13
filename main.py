import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import mysql.connector
import json
import statistics
import numpy as np

def get_data_year(year):
    X = []
    y = []
    year_changed = year.replace("/", "_")
    data = json.load(open('db.json'))
    mydb = mysql.connector.connect(
        host=data["host"],
        user=data["user"],
        password=data["password"],
        port=data["port"],
        database=data["database"],
    )
    mycursor = mydb.cursor(dictionary=True)
    mycursor.execute(f"SELECT AVG(DEF) AS RPDEF FROM Player WHERE SeasonYear = %s AND DEF != 0 GROUP BY Team ORDER BY Team", (year,))
    ratings = mycursor.fetchall()
    for rating in ratings:
        X.append(rating["RPDEF"])

    team_defenses = pd.read_csv('./team-defenses/team_defenses_' + year_changed + '.csv')
    team_defenses = team_defenses.sort_values(by='Team', ascending=True)
    y.extend(team_defenses["DEF"].values.tolist())
    return (X, y)

def get_train_data():
    x_train, y_train = [], []
    for i in range(13, 22):
        season = str(i) + "/" + str(i+1)
        x_temp, y_temp = get_data_year(season)
        x_train.extend(x_temp)
        y_train.extend(y_temp)
    return x_train, y_train

def get_predictions_based_on_old_RPDEF():
    # Get the training data (assuming get_train_data() returns X_train, y_train)
    X_train, y_train = get_train_data()
    X_train = np.array(X_train).reshape(-1, 1)

    # Calculate the number of years in your training data
    num_years = len(X_train) // 30  # Assuming 30 values per year

    # Add the year as a feature to the training data
    years_train = np.array([year for year in range(2013, 2013 + num_years)]).reshape(-1, 1)

    # Duplicate the years to match the number of data points per year
    years_train = np.repeat(years_train, 30, axis=0)

    # Create and train a linear regression model
    model = LinearRegression()
    model.fit(np.hstack((X_train, years_train)), y_train)

    # Get the data for the 2022/23 season
    data = get_data_for_new_season()
    X_test = data[0]
    X_test = np.array(X_test).reshape(-1, 1)
    # Add the year as a feature for the test data
    num_test_samples = len(X_test)
    years_test = np.array([2023] * num_test_samples).reshape(-1, 1)

    # Combine the test data with the year feature
    X_test_with_year = np.hstack((X_test, years_test))

    # Make predictions for the 2022/23 season
    predictions = model.predict(X_test_with_year)

    # Evaluate the model's performance
    y_test = data[1]
    mae = mean_absolute_error(y_test, predictions)

    print("Predictions for season 22/23 based on RPDEF metric")
    print(f"Mean Absolute Error: {mae}")
    teams = data[2]
    teams_predictions = dict(zip(teams, predictions))
    sorted_teams_predictions = dict(sorted(teams_predictions.items(), key=lambda item: item[1]))
    print(sorted_teams_predictions)

def get_data_for_new_season():
    X = []
    teams = []
    data = json.load(open('db.json'))
    mydb = mysql.connector.connect(
        host=data["host"],
        user=data["user"],
        password=data["password"],
        port=data["port"],
        database=data["database"],
    )
    mycursor = mydb.cursor(dictionary=True)
    mycursor.execute(f"SELECT P22.Team AS Team22, AVG(P21.DEF) AS RPDEF_21_22 FROM Player AS P22 JOIN Player AS P21 ON P22.Player = P21.Player WHERE P22.SeasonYear = '22/23' AND P21.SeasonYear = '21/22' AND P21.DEF > 0 GROUP BY P22.Team ORDER BY P22.Team")
    ratings = mycursor.fetchall()
    for rating in ratings:
        X.append(rating["RPDEF_21_22"])
        teams.append(rating["Team22"])

    team_defenses = pd.read_csv('./team-defenses/team_defenses_22_23.csv')
    team_defenses = team_defenses.sort_values(by='Team', ascending=True)
    y = team_defenses["DEF"].values.tolist()
    return [X, y, teams]

def get_trivial_predictions():
    actual_21_22 = get_data_year("21/22")[1]
    predictions_22_23 = actual_21_22
    actual_22_23 = get_data_year("22/23")[1]

    mae = mean_absolute_error(predictions_22_23, actual_22_23)

    print("Trivial Predictions for season 22/23 based on 21/22 season averages")
    print(f"Mean Absolute Error: {mae}")

get_predictions_based_on_old_RPDEF()
get_trivial_predictions()