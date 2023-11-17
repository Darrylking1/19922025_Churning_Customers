import streamlit as st
import pickle
import pandas as pd
import tensorflow as tf

def load_model_and_scaler():
    # Load the Keras model
    loaded_model = tf.keras.models.load_model('mlp_model.h5')

    # Load the scalers using pickle
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    with open('encoder.pkl', 'rb') as encoder_file:
        encoder = pickle.load(encoder_file)

    return loaded_model, scaler, encoder

def user_input():
    st.write("# Customer Churn Prediction")
    st.write("\n Enter customer data")

    tenure = st.number_input("Tenure", value=1, min_value=1)
    monthly_charges = st.number_input("Monthly Charges", value=0.0)
    total_charges = st.number_input("Total Charges", value=0.0)
    senior_citizen = st.selectbox("Senior Citizen", ['No', 'Yes'])
    gender = st.radio("Gender", ['Female', 'Male'])
    partner = st.selectbox("Partner", ['No', 'Yes'])
    dependents = st.selectbox("Dependents", ['No', 'Yes'])
    multiple_lines = st.selectbox("Multiple Lines", ['No phone service', 'No', 'Yes'])
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox("Online Security", ['No', 'Yes', 'No internet service'])
    online_backup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
    device_protection = st.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])
    tech_support = st.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
    streaming_tv = st.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'])
    streaming_movies = st.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'])
    contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
    payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

    data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'SeniorCitizen': senior_citizen,
        'gender': gender,
        'Partner': partner,
        'Dependents': dependents,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method
    }

    return pd.DataFrame(data, index=[0])

def preprocess_user_input(user_input, encoder):
    # Perform necessary preprocessing on the user input
    for column in user_input.select_dtypes(include=['object']).columns:
        if column in encoder:
            user_input[column] = encoder[column].transform([user_input[column]])
    return user_input

def predict_churn(model, preprocessed_data):
    # Make predictions using the provided model
    prediction = model.predict(preprocessed_data)
    return prediction[0]

def run():
    st.set_page_config(
        page_title="Customer Churn Prediction App",
        page_icon="🔄",
    )

    # Load model, scaler, and label encoder
    model, scaler, label_encoder = load_model_and_scaler()

    # User input
    data = user_input()

    # Perform preprocessing on user input
    preprocessed_data = preprocess_user_input(data, label_encoder)

    # Predict customer churn
    predict_button = st.button("Predict Churn")
    if predict_button:
        prediction = predict_churn(model, preprocessed_data)
        if prediction == 0:
            st.success("The customer is predicted to stay.")
        else:
            st.error("The customer is predicted to churn.")

if __name__ == "__main__":
    run()
