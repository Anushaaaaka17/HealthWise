# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from preprocessing import preprocess_data

# Preprocess data
df = preprocess_data()

# Define target and features
target_col = "HeartDisease" if "HeartDisease" in df.columns else "target"
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save model
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, "heart_disease_model.pkl"))
print("💾 Model saved at:", os.path.join(model_dir, "heart_disease_model.pkl"))


import matplotlib.pyplot as plt
import pandas as pd

# Feature importance (ONLY for RandomForest)
importances = model.feature_importances_
feature_names = X.columns

feat_imp = pd.Series(importances, index=feature_names)
feat_imp.sort_values().plot(kind="barh", figsize=(10, 6), color="skyblue")
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()


import joblib

# Save model

print("✅ Model saved as 'heart_disease_model.pkl'")
# Save model and feature names
joblib.dump(model, "models/heart_disease_model.pkl")
joblib.dump(X.columns.tolist(), "models/feature_names.pkl")  # 👈 Add this line
