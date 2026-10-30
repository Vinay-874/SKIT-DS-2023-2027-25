import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load cleaned dataset
df = pd.read_csv("data/processed/nsrdb_cleaned_base.csv")

# 2. Convert timestamp
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# 3. Sort data by time
df = df.sort_values("Timestamp")

# 4. Select input features
features = [
    "GHI",
    "DNI",
    "DHI",
    "Temperature",
    "Relative Humidity",
    "Pressure",
    "Dew Point",
    "Wind Speed",
    "Solar Zenith Angle"
]

# 5. Select target
target = "Target_GHI_NextHour"

# 6. Remove missing values
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# 7. Time-based train-test split
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# 8. Create Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# 9. Train model
print("Training Random Forest Model...")

model.fit(X_train, y_train)

print("Training Completed!")

# 10. Make predictions
y_pred = model.predict(X_test)

# 11. Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nRandom Forest Results")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

import joblib
import os

os.makedirs("models",exist_ok=True)
joblib.dump(model,"models/random_forest_model.pkl")
print("Random Forest Model Saved Succesfully!")
          
          
            #Actual vs predicted Graph
import matplotlib.pyplot as plt
plt.figure(figsize=(12,5)) 
plt.plot(y_test.values[:100], label="Actual GHI")
plt.plot(y_pred[:100], label="Predicted GHI")

plt.xlabel("Test Samples")
plt.ylabel("Solar Radiation (W/m²)")
plt.title("Random Forest - Actual vs Predicted GHI")
plt.legend()
plt.grid(True)

plt.savefig("models/random_forest_prediction.png")
plt.show()

            