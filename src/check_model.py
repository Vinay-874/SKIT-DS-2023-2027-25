
import numpy as np
from pathlib import Path
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

X = np.load(DATA / "X_train_base.npy")
y = np.load(DATA / "y_train_base.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Mean baseline": DummyRegressor(strategy="mean"),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    print(f"\n{name}")
    print(f"MAE:  {mae:.2f} W/m^2")
    print(f"RMSE: {rmse:.2f} W/m^2")
    print(f"R2:   {r2:.4f}")
