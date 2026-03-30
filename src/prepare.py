import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load raw data
df = pd.read_csv('data/raw/iris.csv')

# Split into train/test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save processed data
os.makedirs('data/processed', exist_ok=True)
train_df.to_csv('data/processed/train.csv', index=False)
test_df.to_csv('data/processed/test.csv', index=False)

print(f"Data prepared: {len(train_df)} train samples, {len(test_df)} test samples")