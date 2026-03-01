import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Load data
print("Loading data...")
df = pd.read_csv('german_credit_data.csv')

# Drop the index column if it exists
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

print(f"\nDataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Handle missing values - fill with 'none' for categorical, median for numeric
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna('none')
    else:
        df[col] = df[col].fillna(df[col].median())

print("\n✅ Missing values handled")

# Encode categorical variables
print("\nEncoding categorical variables...")
label_encoders = {}
categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le

# Select features for model
features = [
    'Age',
    'Job', 
    'Credit amount',
    'Duration',
    'Sex_encoded',
    'Housing_encoded',
    'Saving accounts_encoded',
    'Checking account_encoded',
    'Purpose_encoded'
]

# Make sure all features exist
features = [f for f in features if f in df.columns]

print(f"\nFeatures selected: {features}")

# Target variable
if 'Risk' not in df.columns:
    print("\n❌ Error: 'Risk' column not found!")
    print("Please run: python add_risk_labels.py first")
    exit()

X = df[features]
y = df['Risk']

print(f"\n📊 Dataset Summary:")
print(f"Total samples: {len(df)}")
print(f"Features: {len(features)}")
print(f"Good credit (0): {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")
print(f"Bad credit (1): {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")

# Split data (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save everything
print("\n💾 Saving processed data...")
np.save('X_train.npy', X_train_scaled)
np.save('X_test.npy', X_test_scaled)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(features, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("\n✅ SUCCESS! Data prepared and saved!")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print("\nNext step: python train_model.py")