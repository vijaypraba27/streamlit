import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Preprocessing function that ensures correct columns and encoding
def preprocess_input(df):
    df = df.copy()

    # Encode categorical columns manually to match training
    mapping_substance = {'carbon dioxide': 0, 'methane': 1, 'nitrous oxide': 2, 'other GHGs': 3}
    mapping_unit = {'kg/2018 USD, purchaser price': 0, 'kg CO2e/2018 USD, purchaser price': 1}
    mapping_source = {'commodity': 0, 'industry': 1}

    df['Substance'] = df['Substance'].map(mapping_substance)
    df['Unit'] = df['Unit'].map(mapping_unit)
    df['Source'] = df['Source'].map(mapping_source)

    # Add missing columns with default values if needed
    df['Year'] = 2018  # or whatever default year was used during training

    return df

# Load model and scaler 
model = joblib.load('models/LR_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.title("Supply Chain Emissions Prediction")

st.markdown("""
This app predicts **Supply Chain Emission Factors with Margins** based on DQ metrics and other parameters.
""")

# Input form
with st.form("prediction_form"):
    substance = st.selectbox("Substance", ['carbon dioxide', 'methane', 'nitrous oxide', 'other GHGs'])
    unit = st.selectbox("Unit", ['kg/2018 USD, purchaser price', 'kg CO2e/2018 USD, purchaser price'])
    source = st.selectbox("Source", ['commodity', 'industry'])
    supply_wo_margin = st.number_input("Supply Chain Emission Factors without Margins", min_value=0.0)
    margin = st.number_input("Margins of Supply Chain Emission Factors", min_value=0.0)
    dq_reliability = st.slider("DQ Reliability", 0.0, 1.0)
    dq_temporal = st.slider("DQ Temporal Correlation", 0.0, 1.0)
    dq_geo = st.slider("DQ Geographical Correlation", 0.0, 1.0)
    dq_tech = st.slider("DQ Technological Correlation", 0.0, 1.0)
    dq_data = st.slider("DQ Data Collection", 0.0, 1.0)

    submit = st.form_submit_button("Predict")

    if submit:
        input_data = {
            'Substance': substance,
            'Unit': unit,
            'Source': source,
            'Supply Chain Emission Factors without Margins': supply_wo_margin,
            'Margins of Supply Chain Emission Factors': margin,
            'DQ ReliabilityScore of Factors without Margins': dq_reliability,
            'DQ TemporalCorrelation of Factors without Margins': dq_temporal,
            'DQ GeographicalCorrelation of Factors without Margins': dq_geo,
            'DQ TechnologicalCorrelation of Factors without Margins': dq_tech,
            'DQ DataCollection of Factors without Margins': dq_data
        }

        input_df = preprocess_input(pd.DataFrame([input_data]))

        # Check columns match before prediction
        expected_columns = scaler.feature_names_in_  # This attribute works in sklearn >= 1.0
        input_df = input_df[expected_columns]  # Reorder & ensure correct input

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)

        st.success(f"Predicted Supply Chain Emission Factor with Margin: **{prediction[0]:.4f}**")
