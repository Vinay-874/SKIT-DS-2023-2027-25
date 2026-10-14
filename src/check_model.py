
import numpy as np
from pathlib import Path
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

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

    print(f"\n{name}")
    print(f"MAE: {mean_absolute_error(y_test, pred):.2f} W/m²")
    print(f"R²:  {r2_score(y_test, pred):.4f}")
