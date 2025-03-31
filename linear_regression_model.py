import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import zscore

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

# Evaluates model performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Linear Regression Model Results:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R^2: {r2:.2f}")


# Visualizes predictions vs. actual values
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.xlabel("Actual Health Risk Score")
plt.ylabel("Predicted Health Risk Score")
plt.title("Linear Regression: Prediction vs Actual")
plt.grid()
plt.show()

# Optimization technique 1 - Feature Selection - Correlation
correlation_matrix = df[independent_variables + [dependent_variable]].corr()
correlation_with_target = correlation_matrix[dependent_variable].drop(dependent_variable).sort_values(ascending=False)
selected_features = correlation_with_target[abs(correlation_with_target) > 0.2].index.tolist()
print("Selected features based on Correlation Matrix:")
print(selected_features)

# Optimization technique 2 - Outlier Removal
df_selected = df[selected_features + [dependent_variable]]
# Calculates Z-scores
z_scores = np.abs(zscore(df_selected))
# Removes rows with any feature's Z-score > 3
df_filtered = df_selected[(z_scores < 3).mean(axis=1) > 0.9]
print(f"Original rows: {df_selected.shape[0]}, After outlier removal: {df_filtered.shape[0]}")

# Split dataset into training and testing sets
X_opt = df_filtered[selected_features]
y_opt = df_filtered[dependent_variable]

X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(X_opt, y_opt, test_size=0.2, random_state=42)

# Trains the Linear Regression Model
model_opt = LinearRegression()
model_opt.fit(X_train_opt, y_train_opt)

# Makes predictions based on optimized data
y_pred_opt = model_opt.predict(X_test_opt)

# Evaluates model performance
rmse_opt = np.sqrt(mean_squared_error(y_test_opt, y_pred_opt))
r2_opt = r2_score(y_test_opt, y_pred_opt)

print("Linear Regression Model Results:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R^2: {r2:.2f}")


# Visualizes predictions vs. actual values
plt.figure(figsize=(8, 5))
plt.scatter(y_test_opt, y_pred_opt, alpha=0.5, color='blue')
plt.xlabel("Actual Health Risk Score")
plt.ylabel("Predicted Health Risk Score")
plt.title("Optimized Linear Regression: Prediction vs Actual")
plt.grid()
plt.show()
