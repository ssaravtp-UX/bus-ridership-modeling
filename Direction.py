import pandas as pd

df = pd.read_csv("final_bus_model_key_subtypes_with_ons_log.csv")

corr = df.corr(numeric_only=True)["Ons_log"].sort_values(ascending=False)

print(corr)