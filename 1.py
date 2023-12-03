import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the data
df = pd.read_excel('paneer_distribution.xlsx')

# Preprocess the data (example: one-hot encode the 'Day' column)
df_processed = pd.get_dummies(df, columns=['Day'])

# Split the data
X = df_processed.drop(['First Get', 'Second Get'], axis=1)  # Features
y = df_processed[['First Get', 'Second Get']]  # Targets

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train two separate models for 'First Get' and 'Second Get'
model_first_get = LinearRegression()
model_second_get = LinearRegression()

model_first_get.fit(X_train, y_train['First Get'])
model_second_get.fit(X_train, y_train['Second Get'])

# Make predictions
y_pred_first_get = model_first_get.predict(X_test)
y_pred_second_get = model_second_get.predict(X_test)

# Evaluate the models
mse_first_get = mean_squared_error(y_test['First Get'], y_pred_first_get)
mse_second_get = mean_squared_error(y_test['Second Get'], y_pred_second_get)

print(f'MSE for First Get: {mse_first_get}')
print(f'MSE for Second Get: {mse_second_get}')

# Create a DataFrame for the predictions
predictions = X_test.copy()
predictions['First Get Prediction'] = y_pred_first_get
predictions['Second Get Prediction'] = y_pred_second_get

# Reverse the one-hot encoding to get the day names
days = df['Day'].unique()
for day in days:
    predictions[day] = predictions.apply(lambda row: day if row[f'Day_{day}'] == 1 else '', axis=1)
predictions['Day'] = predictions[days].sum(axis=1)

# Plot the predicted paneer amounts for each day
plt.figure(figsize=(10, 5))
plt.bar(predictions['Day'], predictions['First Get Prediction'], label='First Get')
plt.bar(predictions['Day'], predictions['Second Get Prediction'], bottom=predictions['First Get Prediction'], label='Second Get')
plt.xlabel('Day')
plt.ylabel('Predicted Amount of Paneer')
plt.title('Predicted Paneer Amounts on Different Days')
plt.legend()
plt.show()

# Plot the average predicted paneer amounts for First Get vs. Second Get
plt.figure(figsize=(10, 5))
plt.bar(['First Get', 'Second Get'], [predictions['First Get Prediction'].mean(), predictions['Second Get Prediction'].mean()])
plt.xlabel('Get')
plt.ylabel('Average Predicted Amount of Paneer')
plt.title('Average Predicted Paneer Amounts for First Get vs. Second Get')
plt.show()

# Find out on which day students get the most paneer
predictions['Total Paneer Prediction'] = predictions['First Get Prediction'] + predictions['Second Get Prediction']
day_with_most_paneer = predictions.groupby('Day')['Total Paneer Prediction'].mean().idxmax()

print(f"Students get the most paneer on: {day_with_most_paneer}")
