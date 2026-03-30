import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

train_df = pd.read_csv('data/processed/train.csv')
X, y = train_df.drop('target', axis=1), train_df['target']
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')