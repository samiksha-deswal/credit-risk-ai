import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Load data
df = pd.read_csv('german_credit_data.csv')

# Handle missing values
df = df.dropna()  # Simple approach: remove rows with missing data

# Create binary target: 1 = Good Risk, 0 = Bad Risk
df['Risk'] = df['Risk'].map({'good': 0, 'bad': 1})

# Select features for our model
features = ['Age', 'Credit amount', 'Duration']

# Prepare X (features) and y (target)
X = df[features]
y = df['Risk']

# Split into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize the data (makes all numbers on similar scale)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save everything for later use
np.save('X_train.npy', X_train_scaled)
np.save('X_test.npy', X_test_scaled)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Data prepared successfully!")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")