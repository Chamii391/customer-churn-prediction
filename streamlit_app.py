import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import os
import numpy as np

# --- Configuration ---
KERAS_MODEL_FILE = 'employee_attrition_model.h5'
PREPROCESSOR_FILE = 'preprocessor.joblib'

# --- 1. Load Model and Preprocessor Safely ---
@st.cache_resource
def load_keras_model(model_path):
    """Loads the trained Keras model (.h5)."""
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading Keras model '{model_path}': {e}")
            st.stop()
    else:
        st.error(f"Keras model file '{model_path}' not found. Please check your folder.")
        st.stop()

@st.cache_resource
def load_preprocessor(preprocessor_path):
    """Loads the scikit-learn preprocessor (StandardScaler) using joblib."""
    if os.path.exists(preprocessor_path):
        try:
            # The preprocessor is expected to be the fitted StandardScaler
            preprocessor = joblib.load(preprocessor_path)
            return preprocessor
        except Exception as e:
            st.error(f"Error loading preprocessor '{preprocessor_path}': {e}")
            st.stop()
    else:
        st.error(f"Preprocessor file '{preprocessor_path}' not found. Please check your folder.")
        st.stop()

# Load the components
keras_model = load_keras_model(KERAS_MODEL_FILE)
preprocessor = load_preprocessor(PREPROCESSOR_FILE)


# --- CRITICAL: Feature Order and Label Encoding Mappings ---

# Feature Order (MUST match the order provided in the error message)
FEATURE_ORDER = [
    'tenure', 'TotalCharges', 'MonthlyCharges', 'Contract', 'PaymentMethod', 
    'TechSupport', 'OnlineSecurity', 'gender', 'InternetService', 'OnlineBackup'
]

# Label Encoding Mappings (CRITICAL: Must match the integers used during training)
# These are inferred based on alphabetical sorting (the default for LabelEncoder)
LE_MAPPINGS = {
    'gender': {'Female': 0, 'Male': 1},
    # Contract: 'Month-to-month'(0), 'One year'(1), 'Two year'(2)
    'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
    # PaymentMethod: 'Bank transfer'(0), 'Credit card'(1), 'Electronic check'(2), 'Mailed check'(3)
    'PaymentMethod': {'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1, 'Electronic check': 2, 'Mailed check': 3},
    # InternetService: 'DSL'(0), 'Fiber optic'(1), 'No'(2)
    'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
    # 3-Class features: 'No'(0), 'No internet service'(1), 'Yes'(2)
    'OnlineSecurity': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'OnlineBackup': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'TechSupport': {'No': 0, 'No internet service': 1, 'Yes': 2},
}


# --- 2 & 3. Streamlit Layout and Input Widgets ---

st.set_page_config(page_title="Teleco Churn Prediction Pipeline")
st.title("📞 Teleco Customer Churn Prediction Pipeline")
st.markdown("---")

col1, col2 = st.columns(2)

# Input Widgets
with col1:
    st.header("Service & Financial Details")
    tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=24, step=1)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.0, step=1.0, format="%.2f")
    total_charges = st.text_input("Total Charges ($)", value="1000.00", help="Must be a valid number. Input 0 or handle missing logic if needed.") 
    contract = st.selectbox("Contract Type", options=list(LE_MAPPINGS['Contract'].keys()), index=0)
    payment_method = st.selectbox("Payment Method", options=list(LE_MAPPINGS['PaymentMethod'].keys()), index=2)

with col2:
    st.header("Customer & Service Profile")
    gender = st.selectbox("Gender", options=list(LE_MAPPINGS['gender'].keys()), index=0)
    internet_service = st.selectbox("Internet Service", options=list(LE_MAPPINGS['InternetService'].keys()), index=1)
    online_security = st.selectbox("Online Security", options=list(LE_MAPPINGS['OnlineSecurity'].keys()), index=0)
    online_backup = st.selectbox("Online Backup", options=list(LE_MAPPINGS['OnlineBackup'].keys()), index=0)
    tech_support = st.selectbox("Tech Support", options=list(LE_MAPPINGS['TechSupport'].keys()), index=0)


# --- 4. Prediction Button ---
st.markdown("---")
if st.button("Predict Churn", type="primary"):
    
    try:
        # --- CRITICAL FIX: Convert strings to numerical/Label Encoded values ---
        
        # 1. TotalCharges handling (Convert to float, handling potential empty strings if user left it blank)
        total_charges_val = float(total_charges) if total_charges.strip() else 0.0

        # 2. Apply Label Encoding to categorical inputs
        input_data = {
            'tenure': float(tenure),
            'TotalCharges': total_charges_val, # Now a float
            'MonthlyCharges': float(monthly_charges),
            'Contract': LE_MAPPINGS['Contract'][contract],
            'PaymentMethod': LE_MAPPINGS['PaymentMethod'][payment_method],
            'TechSupport': LE_MAPPINGS['TechSupport'][tech_support],
            'OnlineSecurity': LE_MAPPINGS['OnlineSecurity'][online_security],
            'gender': LE_MAPPINGS['gender'][gender],
            'InternetService': LE_MAPPINGS['InternetService'][internet_service],
            'OnlineBackup': LE_MAPPINGS['OnlineBackup'][online_backup]
        }
        
        # --- 5. Create DataFrame with ONLY numerical data and the CORRECT order ---
        # The DataFrame must contain numerical values for ALL 10 features.
        numerical_input_df = pd.DataFrame([input_data], columns=FEATURE_ORDER)
        
        # Step 1: Preprocess the numerical/encoded data using the fitted StandardScaler
        # This will scale the 10 numerical features.
        processed_data = preprocessor.transform(numerical_input_df)
        
        # Step 2: Make Prediction using the Keras model
        probability_churn = keras_model.predict(processed_data, verbose=0)[0][0]
        
        # Convert probability to a binary prediction (0 or 1)
        prediction = np.round(probability_churn)
        
        # --- 6. Display the Final Prediction ---
        st.subheader("Prediction Result")
        
        if prediction == 1:
            st.error("❌ CHURN PREDICTED: This customer is highly likely to leave.")
        else:
            st.success("✅ NO CHURN PREDICTED: This customer is likely to stay.")
            
        churn_proba_percent = probability_churn * 100
        st.markdown(f"**Confidence:** {churn_proba_percent:.2f}% probability of Churn.")
            
    except ValueError:
        st.error("Input Error: Ensure all numerical fields (Tenure, Monthly Charges, Total Charges) are valid numbers.")
        st.stop()
    except KeyError as e:
        st.error(f"Mapping Error: A categorical value (e.g., '{e.args[0]}') does not match the hardcoded Label Encoding options. Please verify the mappings.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction. Error: {e}")