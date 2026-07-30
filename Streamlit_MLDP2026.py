import joblib
import streamlit as st
import numpy as np
import pandas as pd

## Page name
st.set_page_config(
    page_title="Heart Disease Risk Profiler",
    layout="wide",
)


model = joblib.load("heart_disease_rf_model.pkl")


## Create a sidebar panel on the left side of the screen
with st.sidebar:
    ## Add a title to the sidebar
    st.title("Medical Input Column Names")
    ## Add brief instruction text for the user
    st.write("Use this Dictionary to understand the clinical parameters.")
    ## Create an expandable dropdown for Maximum Heart Rate    
    with st.expander("Resting Blood Pressure (trestbps)"):
        st.markdown("Resting blood pressure (in mm Hg) upon admission to the hospital.")
        
    with st.expander("Serum Cholesterol (chol)"):
        st.markdown("Serum cholesterol measurement in mg/dl.")
        
    with st.expander("Fasting Blood Sugar (fbs)"):
        st.markdown("Indicates if the patient's fasting blood sugar is greater than 120 mg/dl, which can be a sign of diabetes.")
        
    with st.expander("Resting ECG (restecg)"):
        st.markdown("Results of the resting electrocardiogram.\n- **Normal:** Normal finding.\n- **LV Hypertrophy:** Probable or definite left ventricular hypertrophy.\n- **ST-T Abnormality:** ST-T wave abnormality.")

    with st.expander("Maximum Heart Rate (thalch)"):
        ## Provide the definition inside the expander
        st.markdown("The highest heart rate a person achieves during a physical stress test.")
        
    ## Create an expandable dropdown for Exercise-Induced Angina
    with st.expander("Exercise-Induced Angina (exang)"):
        ## Provide the definition inside the expander
        st.markdown("Whether physical exercise triggered chest pain (angina).")
        
    ## Create an expandable dropdown for ST Segment Slope
    with st.expander("ST Segment Slope (slope)"):
        ## Provide the definitions using bullet points
        st.markdown("How the heart's electrical wave behaves on an ECG machine right at the peak of exercise.\n\n- **Upsloping:** Normal, healthy response.\n- **Flat:** Abnormal, indicates the heart is starting to starve for oxygen.\n- **Downsloping:** Abnormal, strong warning sign of blocked arteries.")
        
    ## Create an expandable dropdown for Chest Pain Type
    with st.expander("Chest Pain Type (cp)"):
        ## Provide the definitions using bullet points
        st.markdown("- **Typical Angina:** Classic chest pain caused by reduced blood flow.\n- **Atypical Angina:** Chest pain not strictly matching typical angina.\n- **Non-anginal:** Chest pain not related to the heart.\n- **Asymptomatic:** No chest pain.")

    with st.expander("ST Depression (oldpeak)"):
        st.markdown("ST depression induced by exercise relative to rest. An abnormality check measured via ECG.")
        
    with st.expander("Number of Major Vessels (ca)"):
        st.markdown("Number of major blood vessels (0-3) colored by a medical scan called fluoroscopy.")
        
    with st.expander("Thalassemia (thal)"):
        st.markdown("A genetic blood condition.\n- **Normal:** Normal blood flow.\n- **Fixed Defect:** No blood flow in some part of the heart.\n- **Reversable Defect:** Blood flow is observed but not normal.")


## Set the main title of the web application
st.title("Heart Disease Prediction Model")
# Provide instructions for the user
st.write("Adjust the patient clinical measurements below to predict heart disease risk:")

## Create interactive sliders for continuous numerical data. 
## Defaults are set to healthy to ensure a 'Low Risk' initial load.
age = st.slider("Age in years (age)", min_value=28, max_value=77, value=45)
trestbps = st.slider("Resting Blood Pressure in mmHg (trestbps)", min_value=80, max_value=200, value=120)
chol = st.slider("Serum Cholesterol in mg/dl (chol)", min_value=0, max_value=603, value=200)
thalch = st.slider("Maximum Heart Rate Achieved (thalch)", min_value=60, max_value=202, value=170)
oldpeak = st.slider("ST Depression (oldpeak)", min_value=-2.6, max_value=6.2, value=0.0, step=0.1)

## Create interactive dropdown boxes for categorical data
sex = st.selectbox("Gender (sex)", ["Female", "Male"])
cp = st.selectbox("Chest Pain Type (cp)", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
restecg = st.selectbox("Resting ECG (restecg)", ["normal", "lv hypertrophy", "st-t abnormality"])
slope = st.selectbox("ST Segment Slope (slope)", ["upsloping", "flat", "downsloping"])

## Values are passed as floats (.0) to perfectly match the raw dataset formatting
## Without this, the model will flag unseen feature names like 'ca_1' instead of 'ca_1.0'
ca = st.selectbox("Number of Major Vessels (ca)", [0.0, 1.0, 2.0, 3.0])

thal = st.selectbox("Thalassemia (thal)", ["normal", "fixed defect", "reversable defect"])

## Create Yes/No dropdowns for boolean (True/False) fields to make it user-friendly
fbs_ui = st.selectbox("Fasting Blood Sugar > 120 mg/dl? (fbs)", ["No", "Yes"])
exang_ui = st.selectbox("Exercise-Induced Angina? (exang)", ["No", "Yes"])

## Convert the Yes/No answers back to "True"/"False" strings so the model understands them
fbs = "True" if fbs_ui == "Yes" else "False"
exang = "True" if exang_ui == "Yes" else "False"


## Create a button that runs the prediction code only when clicked
if st.button("Predict Heart Disease Risk", type="primary"):

    input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal]]
    ## Defining the exact column names matching the training dataset
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    ## Create a Pandas DataFrame using the list and column names
    df_input = pd.DataFrame(input_data, columns=columns)

    ## One-Hot Encoding
    ## Define which columns are categorical
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    ## Apply get_dummies to convert categories into binary (0 or 1) columns, dropping the first to avoid multicollinearity
    df_input = pd.get_dummies(df_input, columns=categorical_cols, drop_first=True)


    ## Create an exact list of the 20 columns the Random Forest model expects to see
    ## Note the .0 on the ca variables to align with the original trained model schema
    model_columns = [
        'age', 
        'trestbps', 
        'chol', 
        'thalch', 
        'oldpeak', 
        'sex_Male', 
        'cp_atypical angina', 
        'cp_non-anginal', 
        'cp_typical angina', 
        'fbs_True', 
        'restecg_normal', 
        'restecg_st-t abnormality', 
        'exang_True', 
        'slope_flat', 
        'slope_upsloping', 
        'ca_1.0', 
        'ca_2.0', 
        'ca_3.0', 
        'thal_normal', 
        'thal_reversable defect'
    ]
    
    ## Force the user input DataFrame to match the model_columns list exactly, filling missing dummy columns with 0
    df_input = df_input.reindex(columns=model_columns, fill_value=0)


    ## Generate Prediction
    ## Feed the processed DataFrame into the trained model to predict the class (0 or 1)
    prediction = model.predict(df_input)[0]
    ## Feed the processed DataFrame into the trained model to get the confidence percentage
    prediction_proba = model.predict_proba(df_input)[0]


    ## Display Results
    ## Draw a visual horizontal line to separate the button from the results
    st.markdown("---")
    
    ## If the model predicts 1 (Heart Disease)
    if prediction == 1:
        ## Display a red error box warning the user
        st.error("Prediction: High Risk of Heart Disease Detected")
        ## Print the exact confidence percentage for class 1
        st.write(f"Confidence Score: {prediction_proba[1] * 100:.2f}%")
        
    ## If the model predicts 0 (Healthy)
    else:
        ## Display a green success box
        st.success("Prediction: Low Risk / No Heart Disease Detected\n(Low Risk does not mean no risk)")
        ## Print the exact confidence percentage for class 0
        st.write(f"Confidence Score: {prediction_proba[0] * 100:.2f}%")

    
    ## Clinical Advice Section for Doctors/Nurses
    st.markdown("---")
    st.markdown("### Clinical Advice")
    
    advice_list = []
    risk_count = 0
    
    ## Blood Pressure Check
    if trestbps > 120:
        advice_list.append("- **High Blood Pressure:** Resting BP is above 120 mmHg. Eat less salt, exercise daily and use blood pressure medication correctly..")
        risk_count += 1
        
    ## Cholesterol Check
    if chol > 200:
        advice_list.append("- **High Cholesterol:** Serum cholesterol is above 200 mg/dl. Eat less saturated fat, increase soluble fiber intake, exercise daily and evaluate the need for cholesterol-lowering medication.")
        risk_count += 1
        
    ## Fasting Blood Sugar Check
    if fbs == "True":
        advice_list.append("- **High Blood Sugar:** Fasting blood sugar is over 120 mg/dl. This suggests prediabetes or diabetes. Recommend an HbA1c(Blood) test and diet adjustments.")
        risk_count += 1
        
    ## Max Heart Rate Check
    estimated_max_hr = 220 - age
    if thalch < (estimated_max_hr * 0.7):
        advice_list.append(f"- **Low Maximum Heart Rate:** Heart rate during exercise ({thalch} bpm) was lower than expected for their age. Verify if they are on beta-blockers(MEDICNE that LOWERS BLOOD PRESSURE and Slows HEART RATE) or assess their cardiovascular fitness.")

    ## Display compounded risk warning if multiple flags are triggered
    if risk_count >= 1:
        st.error(f" **Compounded Risk Warning:** The patient presents {risk_count} primary metabolic risk factors (High BP, Cholesterol, or Blood Sugar). Combined risk factors exponentially increase the likelihood of a cardiovascular event. Prioritize comprehensive lifestyle and close medical follow-up.")

    ## Display the specific clinical advice notes
    if len(advice_list) > 0:
        for advice in advice_list:
            st.warning(advice)
    else:
        st.success("- All major clinical parameters (Cholesterol, Blood Pressure, Blood Sugar) are currently within standard healthy ranges. Encourage the patient to maintain their current healthy habits.")
