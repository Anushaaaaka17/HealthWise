# model_building.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
df = pd.read_csv(
    "C:\\Users\\anush\\OneDrive\\Desktop\\HealthWise-Project\\data\\cleaned_heart.csv"
)  # Use your cleaned dataset

# ✅ Display columns for verification
print("Columns:", df.columns.tolist())

# ✅ Define features and target
target_col = "HeartDisease"
X = df.drop(columns=[target_col])
y = df[target_col]

# ✅ Encode categorical columns (if any)
X = pd.get_dummies(X, drop_first=True)

# After X = pd.get_dummies(...)
import joblib

joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")


# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Model: Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# ✅ Predictions
y_pred = model.predict(X_test_scaled)

# ✅ Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get predicted probabilities for class 1 (HeartDisease = 1)
y_probs = model.predict_proba(X_test_scaled)[:, 1]

# ROC Curve and AUC Score
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plotting ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Print AUC
print(f"\n🎯 AUC Score: {roc_auc:.2f}")
