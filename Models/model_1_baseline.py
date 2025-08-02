# importing necessary modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from config_local import FILE_PATH
import seaborn as sns

# Load the dataset
dataset = pd.read_csv(FILE_PATH +'\\kc_house_data.csv')

# Select features
features = ['bedrooms','bathrooms','sqft_living', 'floors']
target = 'price'

X = dataset[features]
y = dataset[target]

# Train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the  model
model_1_baseline = LinearRegression()
model_1_baseline.fit(X_train, y_train)

# Make predictions
y_pred = model_1_baseline.predict(X_test)

# Evaluating the model
MSE = mean_squared_error(y_test, y_pred)
R2 = r2_score(y_test, y_pred)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R^2 Score:", r2_score(y_test, y_pred))

# Visualizing the results
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices (Baseline Model")
plt.show()