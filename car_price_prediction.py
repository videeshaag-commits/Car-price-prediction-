import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Sample dataset (you can replace with real dataset later)
data = {
    "Year": [2010, 2012, 2015, 2018, 2020, 2011, 2016, 2019],
    "Km_Driven": [50000, 30000, 40000, 20000, 10000, 60000, 35000, 15000],
    "Fuel_Type": ["Petrol", "Diesel", "Petrol", "Diesel", "Petrol", "Diesel", "Petrol", "Diesel"],
    "Price": [3.5, 4.0, 5.5, 7.0, 9.0, 3.0, 6.0, 8.5]
}

df = pd.DataFrame(data)

# Convert categorical data (Fuel_Type) to numeric
df = pd.get_dummies(df, drop_first=True)

# Features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# Example prediction
sample = np.array([[2019, 20000, 1]])  # Year, Km_Driven, Fuel_Type_Diesel
predicted_price = model.predict(sample)
print("Predicted Price:", predicted_price[0])
