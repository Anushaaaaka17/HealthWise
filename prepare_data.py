# prepare_data.py

import pandas as pd
import os

# Load data
raw_path = os.path.join("data", "heart.csv")
df = pd.read_csv(
    "C:\\Users\\anush\\OneDrive\\Desktop\\HealthWise-Project\\data\\heart.csv"
)

# Show basic info
print("🔹 Raw Data Shape:", df.shape)
print("🔹 Columns:", df.columns.tolist())
print("\n🧹 Missing Values:\n", df.isnull().sum())

# Save preview (optional)
preview_path = os.path.join("data", "preview.csv")
df.head().to_csv(preview_path, index=False)
print("📄 Preview saved at:", preview_path)
