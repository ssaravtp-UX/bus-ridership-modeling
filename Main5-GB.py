import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("final_bus_model_key_subtypes_with_ons_log.csv")

df["Ons_log"] = pd.to_numeric(df["Ons_log"], errors="coerce")
df = df.dropna(subset=["Ons_log"])

X = df.drop(columns=["stop_name", "Ons", "Ons_log"], errors="ignore")
y = df["Ons_log"]

X = pd.get_dummies(X, dummy_na=True)
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

gb = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb.fit(X_train, y_train)
pred = gb.predict(X_test)

print("Rows used:", len(df))
print("Number of features:", X.shape[1])

print("\nGradient Boosting")
print("R2:", r2_score(y_test, pred))
print("MAE:", mean_absolute_error(y_test, pred))

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": gb.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 20 Important Features:")
print(importance.head(20))