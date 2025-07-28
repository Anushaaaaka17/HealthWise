import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load cleaned data
df = pd.read_csv(
    "C:\\Users\\anush\\OneDrive\\Desktop\\HealthWise-Project\\data\\cleaned_heart.csv"
)

# Create output folder if not exists
os.makedirs("outputs/eda_plots", exist_ok=True)

# 1. Target variable distribution
plt.figure(figsize=(6, 4))

plt.title("Heart Disease Distribution")
plt.savefig("outputs/eda_plots/heart_disease_distribution.png")
plt.close()

# 2. Numerical feature distributions
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop(
    "HeartDisease"
)

for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30, color="skyblue")
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(f"outputs/eda_plots/{col}_distribution.png")
    plt.close()

# 3. Categorical vs Target (HeartDisease)
categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col, hue="HeartDisease", palette="Set1")
    plt.title(f"{col} vs Heart Disease")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"outputs/eda_plots/{col}_vs_heart_disease.png")
    plt.close()

# 4. Correlation heatmap
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/eda_plots/correlation_matrix.png")
plt.close()

print("✅ EDA completed. All plots saved to 'outputs/eda_plots/'")
