import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load data
df = pd.read_csv("final_bus_model_key_subtypes_with_ons_log.csv")

# Target
df["Ons_log"] = pd.to_numeric(df["Ons_log"], errors="coerce")
df = df.dropna(subset=["Ons_log"])

# Features
X = df.drop(columns=["stop_name", "Ons", "Ons_log"], errors="ignore")
y = df["Ons_log"]

# Convert text columns to dummy variables
X = pd.get_dummies(X, dummy_na=True)
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

print("Rows used:", len(df))
print("Number of features:", X.shape[1])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest model
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

rf.fit(X_train, y_train)
pred = rf.predict(X_test)

print("\nRandom Forest Results")
print("R2:", r2_score(y_test, pred))
print("MAE:", mean_absolute_error(y_test, pred))

# Feature importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 20 Important Features:")
print(importance.head(20))