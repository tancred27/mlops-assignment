"""Print a quick profile of the raw dataset."""
import pandas as pd

df = pd.read_csv("data/raw/heart.csv")

print("Shape:", df.shape)
print("\nColumns and dtypes:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nSummary statistics:")
print(df.describe().T)
print("\nUnique values per column:")
print(df.nunique())
print("\nTarget distribution (last column):")
print(df.iloc[:, -1].value_counts().sort_index())