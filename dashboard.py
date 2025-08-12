import streamlit as st
import pandas as pd
import joblib


@st.cache_resource
def load_model():
    return joblib.load('thyroid_model.pkl')

model = load_model()


ordinal_categories = {
    'Risk': ['Low', 'Intermediate', 'High'],
    'T': ['T1a', 'T1b', 'T2', 'T3a', 'T3b', 'T4a', 'T4b'],
    'N': ['N0', 'N1a', 'N1b'],
    'M': ['M0', 'M1'],
    'Stage': ['I', 'II', 'III', 'IVA', 'IVB'],
    'Response': ['Excellent', 'Biochemical Incomplete', 'Structural Incomplete', 'Indeterminate']
}

nominal_options = {
    'Gender': ['F', 'M'],
    'Smoking': ['No', 'Yes'],
    'Hx Smoking': ['No', 'Yes'],
    'Hx Radiothreapy': ['No', 'Yes'],
    'Focality': ['Uni-Focal', 'Multi-Focal'],
    'Thyroid Function': ['Euthyroid', 'Clinical Hyperthyroidism', 'Clinical Hypothyroidism', 'Subclinical Hypothyroidism'],
    'Physical Examination': ['Multinodular goiter', 'Single nodular goiter-left', 'Single nodular goiter-right', 'Normal', 'Diffuse goiter'],
    'Adenopathy': ['No', 'Right', 'Left', 'Bilateral', 'Extensive', 'Posterior'],
    'Pathology': ['Micropapillary', 'Papillary', 'Follicular', 'Hurthel cell']
}


st.set_page_config(
    page_title="Thyroid Cancer Recurrence Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
    <style>
    .stApp {
        background-color: #1c2526;  /* Dark gray */
        color: #e8ecef;  /* Light text */
    }
    .stButton > button {
        background-color: #007bff;  /* Blue button */
        color: #ffffff;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #0056b3;  /* Darker blue on hover */
    }
    .stSelectbox > div, .stNumberInput > div {
        background-color: #2a363b;  /* Slightly lighter dark */
        border: 1px solid #495057;
        border-radius: 4px;
        color: #e8ecef;
    }
    .prediction-yes {
        color: #dc3545;  /* Red for Yes */
        font-weight: bold;
        font-size: 1.2em;
    }
    .prediction-no {
        color: #28a745;  /* Green for No */
        font-weight: bold;
        font-size: 1.2em;
    }
    .metric-card {
        background-color: #2a363b;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)


st.title("🩺 Thyroid Cancer Recurrence Predictor")
st.subheader("AI-Powered Risk Assessment for Thyroid Cancer Survivors")
st.markdown("Input patient details to predict recurrence risk using a high-accuracy Random Forest model.")


with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    - Fill in patient details in the 'Predict' tab.
    - Refer to 'Field Explanations' for guidance.
    - Click 'Predict' to view results.
    - Metrics reflect model performance.
    """)

    st.header("📊 Model Performance")
    metrics = [
        ("Accuracy", "97.39%"),
        ("Precision", "100%"),
        ("Recall", "90.63%"),
        ("F1 Score", "95.08%"),
        ("ROC-AUC", "99.51%")
    ]
    for label, value in metrics:
        st.markdown(f"<div class='metric-card'><b>{label}</b>: {value}</div>", unsafe_allow_html=True)
    
    st.header("ℹ️ About")
    st.write("This tool uses a Random Forest model trained on thyroid cancer data. Consult a clinician for medical decisions.")

tab1, tab2 = st.tabs([" Predict", " Field Explanations"])

with tab1:
    with st.form(key="prediction_form"):
        st.header("Patient Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Demographics")
            age = st.number_input("Age", min_value=0, max_value=120, value=30, help="Patient's age in years (0-120).")
            gender = st.selectbox("Gender", options=nominal_options['Gender'], help="Patient's gender (F: Female, M: Male).")
            smoking = st.selectbox("Smoking", options=nominal_options['Smoking'], help="Current smoking status.")
            hx_smoking = st.selectbox("Hx Smoking", options=nominal_options['Hx Smoking'], help="History of smoking.")

        with col2:
            st.subheader("Clinical History")
            hx_radiothreapy = st.selectbox("Hx Radiothreapy", options=nominal_options['Hx Radiothreapy'], help="History of radiotherapy.")
            thyroid_function = st.selectbox("Thyroid Function", options=nominal_options['Thyroid Function'], help="Thyroid status (e.g., Euthyroid = normal).")
            physical_examination = st.selectbox("Physical Examination", options=nominal_options['Physical Examination'], help="Physical exam findings.")
            adenopathy = st.selectbox("Adenopathy", options=nominal_options['Adenopathy'], help="Lymph node enlargement location.")

        with col3:
            st.subheader("Pathology & Staging")
            pathology = st.selectbox("Pathology", options=nominal_options['Pathology'], help="Type of thyroid cancer.")
            focality = st.selectbox("Focality", options=nominal_options['Focality'], help="Cancer focality (single or multiple sites).")
            risk = st.selectbox("Risk", options=ordinal_categories['Risk'], help="Recurrence risk (Low < Intermediate < High).")
            t = st.selectbox("T", options=ordinal_categories['T'], help="Tumor size/invasion (T1a = smallest, T4b = largest).")
            n = st.selectbox("N", options=ordinal_categories['N'], help="Lymph node involvement (N0 = none, N1b = most).")
            m = st.selectbox("M", options=ordinal_categories['M'], help="Metastasis (M0 = none, M1 = present).")
            stage = st.selectbox("Stage", options=ordinal_categories['Stage'], help="Cancer stage (I = earliest, IVB = advanced).")
            response = st.selectbox("Response", options=ordinal_categories['Response'], help="Treatment response (Excellent = best).")

        submit = st.form_submit_button("🔮 Predict Recurrence")

    if submit:
        input_data = pd.DataFrame({
            'Age': [age], 'Gender': [gender], 'Smoking': [smoking], 'Hx Smoking': [hx_smoking],
            'Hx Radiothreapy': [hx_radiothreapy], 'Thyroid Function': [thyroid_function],
            'Physical Examination': [physical_examination], 'Adenopathy': [adenopathy],
            'Pathology': [pathology], 'Focality': [focality], 'Risk': [risk], 'T': [t],
            'N': [n], 'M': [m], 'Stage': [stage], 'Response': [response]
        })

        with st.spinner("Processing patient data..."):
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
        st.header("Prediction Results")
        st.markdown(f"<p class='prediction-{'yes' if prediction == 'Yes' else 'no'}'>Recurrence Prediction: {prediction}</p>", unsafe_allow_html=True)
        st.write(f"**Probability of Recurrence**: {probability:.2%}")
        st.subheader("Recurrence Probability")
        st.progress(probability)
        st.caption("Higher values indicate greater recurrence risk.")

with tab2:
    st.header("Input Field Explanations")
    st.write("Understand each input field to ensure accurate data entry.")
    
    st.subheader("Demographics")
    st.markdown("""
    - **Age**: Patient's age at diagnosis (0-120 years). Example: 45.
    - **Gender**: F (Female) or M (Male). No order.
    - **Smoking**: Current smoking status (No, Yes).
    - **Hx Smoking**: History of smoking (No, Yes).
    """)

    st.subheader("Clinical History")
    st.markdown("""
    - **Hx Radiothreapy**: Previous radiotherapy (No, Yes).
    - **Thyroid Function**: Thyroid status (e.g., Euthyroid = normal function).
    - **Physical Examination**: Exam findings (e.g., Multinodular goiter).
    - **Adenopathy**: Lymph node enlargement (No, Right, Left, etc.).
    """)

    st.subheader("Pathology & Staging")
    st.markdown("""
    - **Pathology**: Cancer type (Micropapillary, Papillary, etc.).
    - **Focality**: Tumor location (Uni-Focal = single, Multi-Focal = multiple).
    - **Risk**: Recurrence risk level (Low < Intermediate < High).
    - **T**: Tumor classification (T1a = smallest to T4b = most invasive).
    - **N**: Nodal involvement (N0 = none to N1b = most extensive).
    - **M**: Metastasis (M0 = none, M1 = present).
    - **Stage**: Cancer stage (I = least severe to IVB = most severe).
    - **Response**: Treatment response (Excellent = best to Indeterminate).
    """)