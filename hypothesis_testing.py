import pandas as pd
from scipy.stats import chi2_contingency

# Load the dataset
df = pd.read_csv(
    "C:\\Users\\anush\\OneDrive\\Desktop\\HealthWise-Project\\data\\cleaned_heart.csv"
)

# Ensure lowercase column names for consistency
df.columns = df.columns.str.lower()

# 1. Sex vs Heart Disease
print("\n--- Sex vs Heart Disease (Chi-Square Test) ---")
if "sex" in df.columns and "heartdisease" in df.columns:
    contingency1 = pd.crosstab(df["sex"], df["heartdisease"])
    chi2_1, p1, dof1, expected1 = chi2_contingency(contingency1)

    print(f"p-value: {p1:.5f}")
    if p1 < 0.05:
        print("✅ Result: Statistically Significant (Reject H₀)")
    else:
        print("❌ Result: Not Statistically Significant (Fail to Reject H₀)")
else:
    print("❌ Columns 'sex' or 'heartdisease' not found in dataset.")

# 2. Fasting Blood Sugar vs Heart Disease
print("\n--- FastingBS vs Heart Disease (Chi-Square Test) ---")
if "fastingbs" in df.columns and "heartdisease" in df.columns:
    contingency2 = pd.crosstab(df["fastingbs"], df["heartdisease"])
    chi2_2, p2, dof2, expected2 = chi2_contingency(contingency2)

    print(f"p-value: {p2:.5f}")
    if p2 < 0.05:
        print("✅ Result: Statistically Significant (Reject H₀)")
    else:
        print("❌ Result: Not Statistically Significant (Fail to Reject H₀)")
else:
    print("❌ Columns 'fastingbs' or 'heartdisease' not found in dataset.")

# 3. Exercise Angina vs Heart Disease
print("\n--- Exercise Angina vs Heart Disease (Chi-Square Test) ---")
if "exerciseangina" in df.columns and "heartdisease" in df.columns:
    contingency3 = pd.crosstab(df["exerciseangina"], df["heartdisease"])
    chi2_3, p3, dof3, expected3 = chi2_contingency(contingency3)

    print(f"p-value: {p3:.5f}")
    if p3 < 0.05:
        print("✅ Result: Statistically Significant (Reject H₀)")
    else:
        print("❌ Result: Not Statistically Significant (Fail to Reject H₀)")
else:
    print("❌ Columns 'exerciseangina' or 'heartdisease' not found in dataset.")
