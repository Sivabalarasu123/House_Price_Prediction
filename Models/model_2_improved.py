# Import necessary modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from config_local import FILE_PATH
import seaborn as sns

# Load dataset
dataset = pd.read_csv(FILE_PATH + "\\kc_house_data.csv")
dataset.info()
#print(dataset.isnull().sum())


# Separate only numeric columns for imputation
numeric_columns= dataset.select_dtypes(include=['int64','float64'])
non_numeric_columns = dataset.select_dtypes(exclude=['int64','float64'])

# Handling the missing values and impute only numeric columns
imputer = SimpleImputer(strategy='mean')
dataset_imputed = pd.DataFrame(imputer.fit_transform(numeric_columns), columns=numeric_columns.columns)

# Split the features and labels using iloc
X = dataset_imputed.iloc[:, :-1] # all columns except the last one
y = dataset_imputed.iloc[:,-1] # only the last column as target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest Regressor
model_2_improved = RandomForestRegressor(n_estimators=100, random_state=42)
model_2_improved.fit(X_train_scaled, y_train)

# Predictions
y_pred = model_2_improved.predict(X_test_scaled)

# Evaluating the model
R2 = r2_score(y_test, y_pred)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R^2 Score:", r2_score(y_test, y_pred))

# Visualizing the results
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices (Improved Model")
plt.show()
