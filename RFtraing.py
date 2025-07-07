import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# 1. Load your custom data
# -------- load  ----------
df = pd.read_csv("crystalData.csv", sep=';')
X = df.drop("Class", axis=1)
y = df["Class"]

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Random Forest
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# 4. Evaluate
print("Accuracy:", clf.score(X_test, y_test))

# 5. Save the model
with open("rf_model.pkl", "wb") as f:
    pickle.dump(clf, f)
