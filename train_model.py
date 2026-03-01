import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix,
    classification_report
)
import pickle

print("🚀 Starting model training...\n")

# Load prepared data
print("📂 Loading training data...")
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Features: {X_train.shape[1]}")

# Create and train the model
print("\n🧠 Training Logistic Regression model...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# Make predictions on test set
print("\n🔮 Making predictions...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# Display results
print("\n" + "="*50)
print("📊 MODEL PERFORMANCE METRICS")
print("="*50)
print(f"✓ Accuracy:  {accuracy:.2%}  - Overall correctness")
print(f"✓ Precision: {precision:.2%}  - When we predict 'bad', how often are we right?")
print(f"✓ Recall:    {recall:.2%}  - Of all actual 'bad' credits, how many did we catch?")
print(f"✓ F1-Score:  {f1:.2%}  - Balance between precision and recall")
print(f"✓ ROC-AUC:   {auc:.2%}  - Model's ability to distinguish between classes")
print("="*50)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n📋 CONFUSION MATRIX:")
print("                  Predicted")
print("                Good    Bad")
print(f"Actual Good     {cm[0][0]:4d}    {cm[0][1]:4d}")
print(f"       Bad      {cm[1][0]:4d}    {cm[1][1]:4d}")

# Interpretation
print("\n💡 INTERPRETATION:")
print(f"• Correctly approved (True Negatives): {cm[0][0]}")
print(f"• Correctly rejected (True Positives): {cm[1][1]}")
print(f"• Wrongly approved (False Negatives): {cm[1][0]} ⚠️ These are risky!")
print(f"• Wrongly rejected (False Positives): {cm[0][1]} (Lost business opportunity)")

# Detailed classification report
print("\n" + "="*50)
print("DETAILED CLASSIFICATION REPORT:")
print("="*50)
print(classification_report(y_test, y_pred, 
                          target_names=['Good Credit (0)', 'Bad Credit (1)']))

# Feature importance (coefficients)
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print("\n🎯 FEATURE IMPORTANCE (Model Coefficients):")
print("="*50)
coefficients = model.coef_[0]
feature_importance = sorted(zip(feature_names, coefficients), 
                           key=lambda x: abs(x[1]), 
                           reverse=True)

for feature, coef in feature_importance:
    direction = "↑ Increases" if coef > 0 else "↓ Decreases"
    print(f"{feature:30s}: {coef:+.4f} {direction} risk")

# Save the trained model
print("\n💾 Saving model...")
with open('credit_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved as 'credit_model.pkl'")

# Business context
print("\n" + "="*50)
print("🎯 BUSINESS IMPACT SUMMARY")
print("="*50)
print(f"With {accuracy:.1%} accuracy and {auc:.2f} AUC:")
print(f"• Out of 100 applications, we correctly assess ~{accuracy*100:.0f}")
print(f"• We catch {recall:.1%} of risky applicants")
print(f"• When we flag as risky, we're right {precision:.1%} of the time")
print("\n🚀 Ready for deployment!")
print("\nNext step: python ai_explainer.py (to test AI explanations)")