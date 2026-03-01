import pandas as pd
import numpy as np

# Load your data
df = pd.read_csv('german_credit_data.csv')

print("Creating risk labels based on applicant characteristics...")

# Create a risk score based on multiple factors
risk_score = 0

# Factor 1: Age (younger = slightly higher risk)
risk_score += (df['Age'] < 25).astype(int) * 15

# Factor 2: High credit amount relative to typical
risk_score += (df['Credit amount'] > df['Credit amount'].median()).astype(int) * 20

# Factor 3: Long duration
risk_score += (df['Duration'] > 36).astype(int) * 25

# Factor 4: Job type (lower job code = higher risk)
risk_score += (df['Job'] <= 1).astype(int) * 15

# Factor 5: Housing status
risk_score += (df['Housing'] == 'free').astype(int) * 10

# Factor 6: Checking account status
risk_score += (df['Checking account'].isna()).astype(int) * 20
risk_score += (df['Checking account'] == 'little').astype(int) * 10

# Factor 7: Savings account status  
risk_score += (df['Saving accounts'].isna()).astype(int) * 15
risk_score += (df['Saving accounts'] == 'little').astype(int) * 10

# Add some randomness to make it realistic
np.random.seed(42)
risk_score += np.random.randint(-10, 10, len(df))

# Convert to binary: 0 = good (approved), 1 = bad (rejected)
# Higher score = higher risk = more likely to default
threshold = risk_score.median()
df['Risk'] = (risk_score > threshold).astype(int)

print(f"\n📊 Risk Distribution:")
print(f"Good Credit (0): {(df['Risk'] == 0).sum()} ({(df['Risk'] == 0).sum()/len(df)*100:.1f}%)")
print(f"Bad Credit (1): {(df['Risk'] == 1).sum()} ({(df['Risk'] == 1).sum()/len(df)*100:.1f}%)")

# Save the updated dataset
df.to_csv('german_credit_data.csv', index=False)

print("\n✅ Risk labels added successfully!")
print("You can now run: python prepare_data_v2.py")