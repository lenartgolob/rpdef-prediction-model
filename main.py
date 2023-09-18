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

# def get_average_baskets_prediction():
#     average_baskets_per_season = []
#     for i in range(13, 22):
#         season = str(i) + "_" + str(i+1)
#         team_defenses = pd.read_csv('./team-defenses/team_defenses_' + season + '.csv')
#         average_baskets_per_season.append(statistics.mean(team_defenses["DEF"].values.tolist()))
#     seasons = np.arange(13, 22)  # Represents the seasons from 13/14 to 22/23
#     # Reshape the seasons as a 2D array
#     seasons = np.array(seasons).reshape(-1, 1)
#     # Create a linear regression model
#     model = LinearRegression()
#     # Fit the model to the data
#     model.fit(seasons, average_baskets_per_season)
#     # Predict the value for the 22/23 season
#     predicted_average_22_23_season = model.predict(np.array([[23]]))[0]
#
#     average_train = sum(average_baskets_per_season) / len(average_baskets_per_season)
#     return predicted_average_22_23_season


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
X_test, _ = get_data_year("22/23")
X_test = np.array(X_test).reshape(-1, 1)
# Add the year as a feature for the test data
num_test_samples = len(X_test)
years_test = np.array([2023] * num_test_samples).reshape(-1, 1)

# Combine the test data with the year feature
X_test_with_year = np.hstack((X_test, years_test))

# Make predictions for the 2022/23 season
predictions = model.predict(X_test_with_year)

# Evaluate the model's performance
team_defenses = pd.read_csv('./team-defenses/team_defenses_22_23.csv')
team_defenses = team_defenses.sort_values(by='Team', ascending=True)
y_test = team_defenses["DEF"].values.tolist()
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae}")
print(y_test)
print(predictions)
