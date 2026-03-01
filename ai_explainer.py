import os
from dotenv import load_dotenv

load_dotenv()

def explain_decision(applicant_data, prediction, probability):
    """
    Generate human-friendly explanation using smart rules (NO API NEEDED!)
    
    Args:
        applicant_data: dict with Age, Credit amount, Duration
        prediction: 0 (approved) or 1 (rejected)
        probability: float (0-1) risk score
    
    Returns:
        str: Natural language explanation
    """
    
    age = applicant_data.get('Age', 0)
    amount = applicant_data.get('Credit amount', 0)
    duration = applicant_data.get('Duration', 0)
    
    # Calculate some useful metrics
    monthly_payment = amount / duration if duration > 0 else 0
    
    if prediction == 0:  # APPROVED
        explanation = "✅ **Loan Application Approved!** "
        
        if probability < 0.3:
            explanation += f"Your application demonstrates strong creditworthiness with a low risk score of {probability:.1%}. "
            
            # Highlight positive factors
            positive_factors = []
            if age >= 35:
                positive_factors.append("stable age profile")
            if amount <= 10000:
                positive_factors.append("moderate loan amount")
            if duration <= 36:
                positive_factors.append("reasonable repayment period")
            
            if positive_factors:
                explanation += "Key strengths: " + ", ".join(positive_factors) + ". "
            
            explanation += "We're confident in your ability to manage this credit responsibly."
            
        elif probability < 0.5:
            explanation += f"While your application is approved, your risk score of {probability:.1%} indicates moderate risk. "
            
            # Suggest improvements
            if duration > 36:
                explanation += "Consider a shorter loan term to reduce overall interest. "
            if amount > 15000:
                explanation += "A smaller loan amount might help build a stronger credit history. "
            
            explanation += "Maintain consistent payments to improve future loan terms."
            
        else:  # 0.5 - 0.7 (edge case approved)
            explanation += f"Your application is approved with conditions. Risk score: {probability:.1%}. "
            explanation += f"Monthly payment will be approximately ${monthly_payment:.0f}. "
            explanation += "We recommend setting up automatic payments to ensure consistent repayment history."
    
    else:  # REJECTED
        explanation = f"❌ **Application Not Approved.** With a risk score of {probability:.1%}, we're unable to approve your application at this time. "
        
        # Identify specific concerns
        concerns = []
        recommendations = []
        
        if age < 25:
            concerns.append("limited credit history associated with younger applicants")
            recommendations.append("building 6-12 months of credit history")
        
        if amount > 20000:
            concerns.append(f"high loan amount (${amount:,.0f})")
            recommendations.append("applying for a smaller initial amount ($5,000-$10,000)")
        
        if duration > 48:
            concerns.append(f"extended repayment period ({duration} months)")
            recommendations.append("reducing loan term to 24-36 months")
        
        if amount / duration > 1000:  # High monthly payment
            concerns.append("high monthly payment burden")
            recommendations.append("improving debt-to-income ratio")
        
        # Build the explanation
        if concerns:
            explanation += "**Primary concerns:** " + "; ".join(concerns) + ". "
        
        explanation += "\n\n**Recommendations to improve approval chances:**\n"
        for i, rec in enumerate(recommendations, 1):
            explanation += f"{i}. {rec.capitalize()}\n"
        
        if not recommendations:
            explanation += "1. Consider applying with a co-signer\n"
            explanation += "2. Build credit history with a secured credit card\n"
            explanation += "3. Reduce existing debt obligations\n"
        
        explanation += "\nFeel free to reapply after addressing these factors. We're here to help you succeed."
    
    return explanation


# Test the function
if __name__ == "__main__":
    print("🧪 Testing Smart Rule-Based Explainer (100% FREE!)\n")
    
    # Test Case 1: Strong Approval
    print("=" * 70)
    print("TEST 1: Strong Profile - Low Risk (Should be APPROVED)")
    print("=" * 70)
    test_data_1 = {
        'Age': 45,
        'Credit amount': 3000,
        'Duration': 18
    }
    
    explanation_1 = explain_decision(test_data_1, prediction=0, probability=0.25)
    print(f"📋 Applicant Details: Age {test_data_1['Age']}, ${test_data_1['Credit amount']:,} for {test_data_1['Duration']} months")
    print(f"\n💬 Explanation:\n{explanation_1}\n")
    
    # Test Case 2: Clear Rejection
    print("\n" + "=" * 70)
    print("TEST 2: Risky Profile - High Risk (Should be REJECTED)")
    print("=" * 70)
    test_data_2 = {
        'Age': 22,
        'Credit amount': 25000,
        'Duration': 60
    }
    
    explanation_2 = explain_decision(test_data_2, prediction=1, probability=0.85)
    print(f"📋 Applicant Details: Age {test_data_2['Age']}, ${test_data_2['Credit amount']:,} for {test_data_2['Duration']} months")
    print(f"\n💬 Explanation:\n{explanation_2}\n")
    
    # Test Case 3: Borderline Approval
    print("\n" + "=" * 70)
    print("TEST 3: Moderate Risk - Borderline (APPROVED with conditions)")
    print("=" * 70)
    test_data_3 = {
        'Age': 35,
        'Credit amount': 8000,
        'Duration': 36
    }
    
    explanation_3 = explain_decision(test_data_3, prediction=0, probability=0.48)
    print(f"📋 Applicant Details: Age {test_data_3['Age']}, ${test_data_3['Credit amount']:,} for {test_data_3['Duration']} months")
    print(f"\n💬 Explanation:\n{explanation_3}\n")
    
    # Test Case 4: Young applicant rejection
    print("\n" + "=" * 70)
    print("TEST 4: Young Applicant - Moderate Amount")
    print("=" * 70)
    test_data_4 = {
        'Age': 23,
        'Credit amount': 12000,
        'Duration': 48
    }
    
    explanation_4 = explain_decision(test_data_4, prediction=1, probability=0.72)
    print(f"📋 Applicant Details: Age {test_data_4['Age']}, ${test_data_4['Credit amount']:,} for {test_data_4['Duration']} months")
    print(f"\n💬 Explanation:\n{explanation_4}\n")
    
    print("=" * 70)
    print("✅ All tests complete! No API needed - 100% FREE!")
    print("=" * 70)
    print("\n💡 BENEFITS:")
    print("  • Zero cost - no API fees ever")
    print("  • Instant responses - no API latency")
    print("  • Works offline - no internet dependency")
    print("  • Full control - customize logic as needed")
    print("  • Professional quality - clear, actionable explanations")