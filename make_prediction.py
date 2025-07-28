# make_predictions.py

from preprocessing import preprocess_data
import joblib
import os

# Step 1: Preprocess data
print("🔄 Preprocessing data...")
df_cleaned = preprocess_data()

# Separate features
target_col = "HeartDisease" if "HeartDisease" in df_cleaned.columns else "target"
if target_col in df_cleaned.columns:
    X = df_cleaned.drop(columns=[target_col])
else:
    X = df_cleaned

# Step 2: Load model
model_path = os.path.join("models", "heart_disease_model.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError("Trained model not found. Please run train_model.py first.")

model = joblib.load(model_path)
print("✅ Model loaded.")

# Step 3: Predict
predictions = model.predict(X)
df_cleaned["Predicted_HeartDisease"] = predictions

# Step 4: Save predictions
output_path = os.path.join("data", "predicted_heart.csv")
df_cleaned.to_csv(output_path, index=False)
print("📁 Predictions saved at:", output_path)
