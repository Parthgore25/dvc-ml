import pandas as pd
import joblib
import json

# Load model and test data
model = joblib.load('models/model.pkl')
test_df = pd.read_csv('data/processed/test.csv')
X_test = test_df.drop('target', axis=1)
y_test = test_df['target']

# Evaluate
accuracy = model.score(X_test, y_test)

# Save metrics
with open('metrics.json', 'w') as f:
    json.dump({"accuracy": accuracy}, f)

print(f"Evaluation complete. Accuracy: {accuracy:.4f}")