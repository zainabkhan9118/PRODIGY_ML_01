import pandas as pd

# Load the data
df = pd.read_csv('train.csv')

# Look at the first few rows of the data
print(df.head())

# Select the features we need: square footage, bedrooms, bathrooms
features = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]

# Target value: SalePrice
target = df['SalePrice']

print(features.head())  # Look at the selected features


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

print(f'Training Set Size: {X_train.shape}')
print(f'Testing Set Size: {X_test.shape}')

from sklearn.linear_model import LinearRegression

# Create the Linear Regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

print("Model training completed.")


from sklearn.metrics import mean_squared_error

# Use the model to make predictions on the test data
y_pred = model.predict(X_test)

# Calculate how well the model did (lower number is better)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

import matplotlib.pyplot as plt

# Plot the predicted vs actual house prices
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.show()

