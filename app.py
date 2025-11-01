import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load Model and Encoders
# -------------------------------
@st.cache_resource
def load_resources():
    model = joblib.load('extra_trees_credit_model.pickle')

    encoders = {}
    cols_to_load = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    for col in cols_to_load:
        try:
            encoders[col] = joblib.load(f'{col}_encoder.pickle')
        except FileNotFoundError:
            st.warning(f"⚠️ Encoder file for '{col}' not found.")
    return model, encoders


model, encoders = load_resources()

# -------------------------------
# Streamlit Configuration
# -------------------------------
st.set_page_config(page_title="Credit Risk Predictor", layout="centered")
st.title('💳 Credit Risk Prediction App')
st.markdown('Fill out the form below to predict if the applicant’s credit risk is **Good** or **Bad**.')

# -------------------------------
# Input Form
# -------------------------------
with st.form("applicant_form"):
    st.subheader("Applicant Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', min_value=18, max_value=80, value=30)
    with col2:
        sex = st.selectbox('Sex', ['male', 'female'])
    with col3:
        job = st.number_input('Job (0–3 scale)', min_value=0, max_value=3, value=1)

    col4, col5, col6 = st.columns(3)
    with col4:
        housing = st.selectbox('Housing', ['own', 'rent', 'free'])
    with col5:
        saving_accounts = st.selectbox('Saving accounts', ['little', 'moderate', 'rich', 'quite rich'])
    with col6:
        checking_account = st.selectbox('Checking account', ['little', 'moderate', 'rich'])

    st.subheader("Credit Details")

    col7, col8, col9 = st.columns(3)
    with col7:
        credit_amount = st.number_input('Credit amount', min_value=0, value=1000)
    with col8:
        duration = st.number_input('Duration (months)', min_value=1, value=12)
    with col9:
        purpose = st.selectbox('Purpose', [
            'car', 'radio/TV', 'furniture/equipment', 'business', 'education',
            'repairs', 'domestic appliances', 'vacation/others'
        ])

    submitted = st.form_submit_button('Predict Risk')

# -------------------------------
# Safe Encoder
# -------------------------------
def safe_encode(col_name, raw_value):
    """Encode categorical values safely (lowercase match)."""
    enc = encoders.get(col_name)
    if enc is None:
        return raw_value

    # Convert to lowercase (LabelEncoder trained on lowercase)
    raw_value = str(raw_value).strip().lower()

    try:
        return int(enc.transform([raw_value])[0])
    except ValueError:
        # Fallback: use first known class silently
        return int(enc.transform([enc.classes_[0]])[0])

# -------------------------------
# Prediction Logic
# -------------------------------
if submitted:
    input_data = {
        'Age': [age],
        'Sex': [safe_encode('Sex', sex)],
        'Job': [job],
        'Housing': [safe_encode('Housing', housing)],
        'Saving accounts': [safe_encode('Saving accounts', saving_accounts)],
        'Checking account': [safe_encode('Checking account', checking_account)],
        'Credit amount': [credit_amount],
        'Duration': [duration],
        'Purpose': [safe_encode('Purpose', purpose)]
    }

    input_df = pd.DataFrame(input_data)

    # Ensure columns match model
    if hasattr(model, "feature_names_in_"):
        missing = set(model.feature_names_in_) - set(input_df.columns)
        for col in missing:
            input_df[col] = 0
        input_df = input_df[model.feature_names_in_]

    # Predict
    prediction = model.predict(input_df)[0]

    # Display clean result only
    if prediction == 1:
        st.success('✅ The predicted credit risk is **Good (Low Risk)**')
    else:
        st.error('🚨 The predicted credit risk is **Bad (High Risk)**')
