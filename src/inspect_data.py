import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("data/application_train.csv")

# Dataset structure
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Target distribution
print("\nTarget distribution:")
print(df["TARGET"].value_counts())

print("\nTarget percentages:")
print(df["TARGET"].value_counts(normalize=True) * 100)

# Data quality checks
print("\nDuplicate application IDs:",
      df["SK_ID_CURR"].duplicated().sum())

print("Missing target values:",
      df["TARGET"].isnull().sum())