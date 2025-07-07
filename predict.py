import pandas as pd
import joblib

# --- load model ---
model = joblib.load("rf_model.pkl")

# --- load new samples (same columns, same order) ---
new_data = pd.read_csv("crystalData.csv", sep=';').head(2)
print(new_data.head(2))
X = new_data.drop("Class", axis=1)
y = new_data["Class"]
preds = model.predict(X)

print("Predicted classes:", preds)
