import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

# Loads dataset
file_path = "./data/DQN1_Dataset.csv"
df = pd.read_csv(file_path)

# Converts epoch times to readable date format
df["datetime"] = pd.to_datetime(df["datetimeEpoch"], unit="s")
df["sunrise_time"] = pd.to_datetime(df["sunriseEpoch"], unit="s")
df["sunset_time"] = pd.to_datetime(df["sunsetEpoch"], unit="s")
df["year"] =  df["datetime"].dt.year
df["day"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour

# Shifts time data to beginning of file
df = df[["datetime", "year", "month", "day", "hour", "sunrise_time", "sunset_time"] +
        [col for col in df.columns if col not in ["datetime", "year", "month", "day", "hour", "sunrise_time",
                                                  "sunset_time"]]]

# Sets independent and dependent variables
dependent_variable = "healthRiskScore"
independent_variables = ["tempmax", "tempmin", "temp", "feelslikemax", "feelslikemin", "feelslike", "pm2.5", "no2",
                         "co2", "dew", "humidity", "precip", "precipprob", "precipcover", "windgust", "windspeed",
                         "winddir", "pressure", "cloudcover", "visibility", "solarradiation", "solarenergy", "uvindex",
                         "severerisk", "moonphase", "tempRange", "heatIndex",
                         "severityScore", "dayOfWeek", "isWeekend"]

# Split dataset into training and testing sets
x = df[independent_variables]
y = df[dependent_variable]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Trains the Linear Regression Model
linear_regression = LinearRegression()
linear_regression.fit(x_train, y_train)

# Makes predictions based on test data
y_pred = linear_regression.predict(x_test)
