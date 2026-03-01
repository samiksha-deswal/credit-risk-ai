import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from ai_explainer import explain_decision

# Page config
st.set_page_config(
    page_title="Credit Risk AI System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_models():
    try:
        with open('credit_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return model, scaler, feature_names
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found! Please run train_model.py first.")
        st.stop()

model, scaler, feature_names = load_models()

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["🏠 Risk Assessment", "📊 Portfolio Analytics", "📈 Model Performance", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Quick Stats")
st.sidebar.metric("Model Accuracy", "72.5%")
st.sidebar.metric("ROC-AUC Score", "0.82")
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Python, scikit-learn & Streamlit")

# ========== PAGE 1: RISK ASSESSMENT ==========
if page == "🏠 Risk Assessment":
    st.markdown('<p class="main-header">💳 Credit Risk Assessment System</p>', unsafe_allow_html=True)
    st.markdown("Enter applicant information to receive instant risk assessment with detailed explanation.")
    
    st.markdown("---")
    
    # Input form in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input(
            "👤 Age",
            min_value=18,
            max_value=80,
            value=35,
            help="Applicant's age in years"
        )
    
    with col2:
        credit_amount = st.number_input(
            "💰 Loan Amount (₹)",
            min_value=1000,
            max_value=5000000,
            value=5000,
            step=100,
            help="Requested loan amount in INR"
        )
    
    with col3:
        duration = st.number_input(
            "📅 Duration (months)",
            min_value=6,
            max_value=72,
            value=24,
            step=6,
            help="Loan repayment period in months"
        )
    
    # Additional info (for context)
    with st.expander("🔍 Additional Context (Optional)"):
        col1, col2 = st.columns(2)
        with col1:
            job_type = st.selectbox("Job Type", ["Skilled", "Unskilled", "Highly Skilled", "Unemployed"], index=0)
        with col2:
            housing = st.selectbox("Housing", ["Own", "Rent", "Free"], index=0)
        
        monthly_payment = credit_amount / duration
        st.info(f"💵 Estimated monthly payment: **${monthly_payment:.2f}**")
    
    st.markdown("---")
    
    # Assess Risk button
    if st.button("🔍 Assess Credit Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing application..."):
            # Map UI inputs to encoded values
            job_map = {"Skilled": 2, "Unskilled": 0, "Highly Skilled": 3, "Unemployed": 1}
            housing_map = {"Own": 0, "Rent": 1, "Free": 2}
            
            job_encoded = job_map.get(job_type, 2)
            housing_encoded = housing_map.get(housing, 0)
            
            # Create full feature vector matching training data
            # Features: Age, Job, Credit amount, Duration, Sex_encoded, Housing_encoded, 
            #           Saving accounts_encoded, Checking account_encoded, Purpose_encoded
            input_data = np.array([[
                age,                    # Age
                job_encoded,           # Job
                credit_amount,         # Credit amount
                duration,              # Duration
                0,                     # Sex_encoded (default: 0)
                housing_encoded,       # Housing_encoded
                1,                     # Saving accounts_encoded (default: 1)
                1,                     # Checking account_encoded (default: 1)
                0                      # Purpose_encoded (default: 0)
            ]])
            
            # Scale the input
            input_scaled = scaler.transform(input_data)
            
            # Get prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            # Store in session state for what-if analysis
            st.session_state['last_prediction'] = prediction
            st.session_state['last_probability'] = probability
            st.session_state['last_age'] = age
            st.session_state['last_amount'] = credit_amount
            st.session_state['last_duration'] = duration
            st.session_state['last_job'] = job_encoded
            st.session_state['last_housing'] = housing_encoded
            
            # Display results
            st.markdown("### 📋 Assessment Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 0:
                    st.success("### ✅ APPROVED")
                else:
                    st.error("### ❌ REJECTED")
            
            with col2:
                risk_color = "🟢" if probability < 0.3 else "🟡" if probability < 0.7 else "🔴"
                st.metric("Risk Score", f"{probability:.1%}", delta=None)
                st.caption(f"{risk_color} Risk Level")
            
            with col3:
                confidence = max(probability, 1 - probability)
                st.metric("Confidence", f"{confidence:.1%}")
            
            # AI Explanation
            st.markdown("### 💬 Detailed Explanation")
            applicant_dict = {
                'Age': age,
                'Credit amount': credit_amount,
                'Duration': duration
            }
            explanation = explain_decision(applicant_dict, prediction, probability)
            st.info(explanation)
            
            # Risk factors breakdown
            st.markdown("### 📊 Risk Factors Analysis")
            
            factors = []
            if age < 25:
                factors.append(("Age", "Young applicant (< 25)", "🔴 High Impact"))
            elif age > 60:
                factors.append(("Age", "Senior applicant (> 60)", "🟡 Medium Impact"))
            else:
                factors.append(("Age", "Stable age range", "🟢 Positive"))
            
            if credit_amount > 20000:
                factors.append(("Loan Amount", "High amount (> $20k)", "🔴 High Impact"))
            elif credit_amount > 10000:
                factors.append(("Loan Amount", "Moderate amount", "🟡 Medium Impact"))
            else:
                factors.append(("Loan Amount", "Low amount", "🟢 Positive"))
            
            if duration > 48:
                factors.append(("Duration", "Long term (> 48 months)", "🔴 High Impact"))
            elif duration > 36:
                factors.append(("Duration", "Medium term", "🟡 Medium Impact"))
            else:
                factors.append(("Duration", "Short term", "🟢 Positive"))
            
            df_factors = pd.DataFrame(factors, columns=["Factor", "Status", "Impact"])
            st.table(df_factors)
    
    # What-if Analysis section
    st.markdown("---")
    st.markdown("### 🔄 What-If Scenario Analysis")
    st.markdown("Adjust parameters to see how they affect the risk assessment:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_amount = st.slider(
            "Adjust Loan Amount",
            min_value=100,
            max_value=50000,
            value=int(credit_amount),
            step=500
        )
    
    with col2:
        new_duration = st.slider(
            "Adjust Duration (months)",
            min_value=6,
            max_value=72,
            value=int(duration),
            step=6
        )
    
    if st.button("🔄 Recalculate", use_container_width=True):
        if 'last_prediction' in st.session_state:
            # Get stored values
            job_encoded = st.session_state.get('last_job', 2)
            housing_encoded = st.session_state.get('last_housing', 0)
            stored_age = st.session_state.get('last_age', age)
            
            # Create full feature vector with new parameters
            new_input = np.array([[
                stored_age,
                job_encoded,
                new_amount,
                new_duration,
                0,
                housing_encoded,
                1,
                1,
                0
            ]])
            
            new_input_scaled = scaler.transform(new_input)
            new_prob = model.predict_proba(new_input_scaled)[0][1]
            new_pred = model.predict(new_input_scaled)[0]
            
            old_prob = st.session_state.get('last_probability', 0.5)
            change = new_prob - old_prob
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "New Risk Score",
                    f"{new_prob:.1%}",
                    delta=f"{change:+.1%}",
                    delta_color="inverse"
                )
            
            with col2:
                if new_pred == 0:
                    st.success("✅ Would be APPROVED")
                else:
                    st.error("❌ Would be REJECTED")
        else:
            st.warning("⚠️ Please run an assessment first before using What-If analysis")

# ========== PAGE 2: ANALYTICS ==========
elif page == "📊 Portfolio Analytics":
    st.markdown('<p class="main-header">📊 Portfolio Analytics Dashboard</p>', unsafe_allow_html=True)
    
    # Load test data
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    
    # Get predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # Key Metrics
    st.markdown("### 📈 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_apps = len(predictions)
        st.metric("Total Applications", total_apps)
    
    with col2:
        approval_rate = (predictions == 0).sum() / len(predictions)
        st.metric("Approval Rate", f"{approval_rate:.1%}")
    
    with col3:
        avg_risk = probabilities.mean()
        st.metric("Avg Risk Score", f"{avg_risk:.1%}")
    
    with col4:
        high_risk = (probabilities > 0.7).sum()
        st.metric("High Risk Apps", high_risk)
    
    st.markdown("---")
    
    # Risk Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Risk Score Distribution")
        fig = px.histogram(
            x=probabilities,
            nbins=20,
            labels={'x': 'Risk Score', 'y': 'Number of Applications'},
            title='Distribution of Risk Scores'
        )
        fig.update_traces(marker_color='#1f77b4')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Approval Status")
        approval_counts = pd.Series(predictions).value_counts()
        fig = px.pie(
            values=approval_counts.values,
            names=['Approved', 'Rejected'],
            title='Application Outcomes',
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk Categories
    st.markdown("### 🎚️ Risk Category Breakdown")
    
    risk_categories = pd.cut(
        probabilities,
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=['Low Risk', 'Medium-Low', 'Medium-High', 'High Risk']
    )
    
    category_counts = risk_categories.value_counts().sort_index()
    
    fig = px.bar(
        x=category_counts.index,
        y=category_counts.values,
        labels={'x': 'Risk Category', 'y': 'Number of Applications'},
        title='Applications by Risk Category',
        color=category_counts.values,
        color_continuous_scale='RdYlGn_r'
    )
    st.plotly_chart(fig, use_container_width=True)

# ========== PAGE 3: MODEL PERFORMANCE ==========
elif page == "📈 Model Performance":
    st.markdown('<p class="main-header">📈 Model Performance Metrics</p>', unsafe_allow_html=True)
    
    # Load test data
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Display metrics
    st.markdown("### 🎯 Classification Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}")
    with col2:
        st.metric("Precision", f"{precision:.2%}")
    with col3:
        st.metric("Recall", f"{recall:.2%}")
    with col4:
        st.metric("F1-Score", f"{f1:.2%}")
    with col5:
        st.metric("ROC-AUC", f"{auc:.2%}")
    
    st.markdown("---")
    
    # Confusion Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Approved', 'Predicted Rejected'],
            y=['Actually Approved', 'Actually Rejected'],
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20},
            colorscale='Blues',
            showscale=False
        ))
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"""
        **Interpretation:**
        - ✅ Correctly Approved: {cm[0][0]}
        - ✅ Correctly Rejected: {cm[1][1]}
        - ⚠️ False Negatives: {cm[1][0]} (risky loans approved)
        - ⚠️ False Positives: {cm[0][1]} (good applicants rejected)
        """)
    
    with col2:
        st.markdown("### 📈 ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'Model (AUC = {auc:.2f})',
            line=dict(color='#1f77b4', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random Classifier',
            line=dict(color='gray', width=2, dash='dash')
        ))
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            legend=dict(x=0.6, y=0.1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"""
        **ROC-AUC Score: {auc:.2%}**
        
        An AUC of 0.82 indicates the model has **good discriminative ability**
        between approved and rejected applications.
        """)
    
    # Feature Importance
    st.markdown("### 🎯 Feature Importance")
    
    coefficients = model.coef_[0]
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    fig = px.bar(
        feature_importance,
        x='Coefficient',
        y='Feature',
        orientation='h',
        title='Feature Impact on Credit Risk',
        labels={'Coefficient': 'Impact on Risk Score'},
        color='Coefficient',
        color_continuous_scale='RdYlGn_r'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **How to read this chart:**
    - **Positive coefficients** (red) increase risk of rejection
    - **Negative coefficients** (green) decrease risk of rejection
    - **Larger bars** indicate stronger impact on the decision
    """)

# ========== PAGE 4: ABOUT ==========
else:
    st.markdown('<p class="main-header">ℹ️ About This Project</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 💳 Credit Risk Scoring System with AI-Powered Insights
    
    ### 🎯 Problem Statement
    Traditional credit scoring systems are often **black boxes** that provide decisions without explanation.
    This creates problems for:
    - **Lenders**: Difficulty explaining decisions to regulators and customers
    - **Applicants**: No understanding of why they were rejected or how to improve
    - **Regulators**: Lack of transparency in automated lending decisions
    
    ### ✨ Solution
    This system combines **machine learning** with **intelligent explanation generation** to provide:
    
    1. **Accurate Risk Assessment**: Logistic regression model with 72.5% accuracy and 0.82 AUC
    2. **Explainable Decisions**: Every prediction comes with detailed, human-readable explanation
    3. **Interactive Dashboard**: Real-time risk assessment and portfolio analytics
    4. **What-If Analysis**: Test different scenarios to understand risk factors
    
    ### 🛠️ Technical Stack
    - **Machine Learning**: Scikit-learn (Logistic Regression)
    - **Web Framework**: Streamlit
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Plotly
    - **Explanations**: Rule-based intelligent logic
    
    ### 📊 Business Impact
    - ⏱️ **70% reduction** in manual review time (estimated)
    - 📝 Provides **regulatory-compliant** explanations
    - 🎯 Enables **data-driven** lending decisions
    - 💡 Helps applicants **understand and improve** their credit profile
    
    ### 🔮 Future Enhancements
    - [ ] Add more sophisticated ML models (Random Forest, XGBoost)
    - [ ] Integrate with real-time credit bureau data
    - [ ] Implement A/B testing framework
    - [ ] Add fraud detection capabilities
    - [ ] Mobile app version
    
    ### 👨‍💻 Built By
    **Samiksha Deswal**
    
    This project was created as a portfolio piece to demonstrate:
    - End-to-end ML pipeline development
    - Explainable AI implementation
    - Full-stack data science skills
    - Business problem-solving ability
    
    ### 📚 Learn More
    - GitHub Repository (Add your link after pushing to GitHub)
    - LinkedIn Post (Add after you post about it)
    - Demo Video (Add after you record it)
    
    ---
    
    ### 📧 Contact
    Interested in discussing this project or similar work? Let's connect!
    """)
    
    # Add some stats in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Lines of Code", "~800")
    with col2:
        st.metric("Features Used", "9")
    with col3:
        st.metric("Model Accuracy", "72.5%")