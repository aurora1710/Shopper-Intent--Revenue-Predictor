import streamlit as st
import pandas as pd
import joblib

# Load the saved model and scaler
model = joblib.load('Shopper_Intention_logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Online Shopper Intention Prediction')

st.sidebar.header('User Input Features')

# Collect user input features
def user_input_features():
    Administrative = st.sidebar.slider('Administrative pages visited', 0, 30, 0)
    Administrative_Duration = st.sidebar.slider('Time spent on Administrative pages (sec)', 0.0, 3000.0, 0.0)
    Informational = st.sidebar.slider('Informational pages visited', 0, 30, 0)
    Informational_Duration = st.sidebar.slider('Time spent on Informational pages (sec)', 0.0, 3000.0, 0.0)
    ProductRelated = st.sidebar.slider('ProductRelated pages visited', 0, 300, 0)
    ProductRelated_Duration = st.sidebar.slider('Time spent on ProductRelated pages', 0.0, 30000.0, 0.0)
    BounceRates = st.sidebar.slider('BounceRates', 0.0, 0.2, 0.0)
    ExitRates = st.sidebar.slider('Average exit rate from the pages visited', 0.0, 0.2, 0.0)
    PageValues = st.sidebar.slider('PageValues', 0.0, 400.0, 0.0)
    SpecialDay = st.sidebar.slider('Closeness of visit to a special day', 0.0, 1.0, 0.0, 0.1)
    Month = st.sidebar.selectbox('Month', 1, 12, 1)
    OperatingSystems = st.sidebar.slider('OperatingSystems', 1, 8, 1)
    Browser = st.sidebar.slider('Browser', 1, 13, 1)
    Region = st.sidebar.slider('Region', 1, 9, 1)
    TrafficType = st.sidebar.slider('TrafficType', 1, 20, 1)
    VisitorType = st.sidebar.selectbox('VisitorType', ('New_Visitor', 'Returning_Visitor', 'Other'))
    Weekend = st.sidebar.checkbox('Weekend')

    # Convert categorical features to numerical
    visitor_type_mapping = {'New_Visitor': 0, 'Returning_Visitor': 1, 'Other': 2}
    VisitorType_numeric = visitor_type_mapping[VisitorType]
    Weekend_numeric = 1 if Weekend else 0

    data = {'Administrative': Administrative,
            'Administrative_Duration': Administrative_Duration,
            'Informational': Informational,
            'Informational_Duration': Informational_Duration,
            'ProductRelated': ProductRelated,
            'ProductRelated_Duration': ProductRelated_Duration,
            'BounceRates': BounceRates,
            'ExitRates': ExitRates,
            'PageValues': PageValues,
            'SpecialDay': SpecialDay,
            'Month': Month,
            'OperatingSystems': OperatingSystems,
            'Browser': Browser,
            'Region': Region,
            'TrafficType': TrafficType,
            'VisitorType': VisitorType_numeric,
            'Weekend': Weekend_numeric}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input features')
st.write(input_df)

# Scale the input features
input_scaled = scaler.transform(input_df)

# Make prediction
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

st.subheader('Prediction')
st.write('Revenue' if prediction[0] else 'No Revenue')

st.subheader('Prediction Probability')
st.write(prediction_proba)
