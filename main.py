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
    # Manually encode the given variables with ranks
    senior_citizen_rank = {'No': 0, 'Yes': 1}
    gender_rank = {'Female': 0, 'Male': 1}
    partner_rank = {'No': 0, 'Yes': 1}
    dependents_rank = {'No': 0, 'Yes': 1}
    phone_service_rank = {'No': 0, 'Yes': 1}
    multiple_lines_rank = {'No phone service': 0, 'No': 1, 'Yes': 2}
    internet_service_rank = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
    online_security_rank = {'No': 0, 'Yes': 1, 'No internet service': 2}
    online_backup_rank = {'No': 0, 'Yes': 1, 'No internet service': 2}
    device_protection_rank = {'No': 0, 'Yes': 1, 'No internet service': 2}
    tech_support_rank = {'No': 0, 'Yes': 1, 'No internet service': 2}
    streaming_tv_rank = {'No': 0, 'Yes': 1, 'No internet service': 2}
    streaming_movies_rank = {'No': 0, 'Yes': 1, 'No internet service': 2}
    contract_rank = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    paperless_billing_rank = {'No': 0, 'Yes': 1}
    payment_method_rank = {
        'Bank transfer (automatic)': 0,
        'Credit card (automatic)': 1,
        'Electronic check': 2,
        'Mailed check': 3
    }
    
    # Collect user input for prediction
    tenure = st.number_input("Tenure", value=1, min_value=1)
    monthly_charges = st.number_input("Monthly Charges", value=0.0)
    total_charges = st.number_input("Total Charges", value=0.0)
    senior_citizen = st.selectbox("Senior Citizen", list(senior_citizen_rank.keys()))
    senior_citizen = senior_citizen_rank[senior_citizen]
    gender = st.radio("Gender", list(gender_rank.keys()))
    gender = gender_rank[gender]
    partner = st.selectbox("Partner", list(partner_rank.keys()))
    partner = partner_rank[partner]
    dependents = st.selectbox("Dependents", list(dependents_rank.keys()))
    dependents = dependents_rank[dependents]
    phone_service = st.selectbox("Phone Service", list(phone_service_rank.keys()))
    phone_service = phone_service_rank[phone_service]
    multiple_lines = st.selectbox("Multiple Lines", list(multiple_lines_rank.keys()))
    multiple_lines = multiple_lines_rank[multiple_lines]
    internet_service = st.selectbox("Internet Service", list(internet_service_rank.keys()))
    internet_service = internet_service_rank[internet_service]
    online_security = st.selectbox("Online Security", list(online_security_rank.keys()))
    online_security = online_security_rank[online_security]
    online_backup = st.selectbox("Online Backup", list(online_backup_rank.keys()))
    online_backup = online_backup_rank[online_backup]
    device_protection = st.selectbox("Device Protection", list(device_protection_rank.keys()))
    device_protection = device_protection_rank[device_protection]
    tech_support = st.selectbox("Tech Support", list(tech_support_rank.keys()))
    tech_support = tech_support_rank[tech_support]
    streaming_tv = st.selectbox("Streaming TV", list(streaming_tv_rank.keys()))
    streaming_tv = streaming_tv_rank[streaming_tv]
    streaming_movies = st.selectbox("Streaming Movies", list(streaming_movies_rank.keys()))
    streaming_movies = streaming_movies_rank[streaming_movies]
    contract = st.selectbox("Contract", list(contract_rank.keys()))
    contract = contract_rank[contract]
    paperless_billing = st.selectbox("Paperless Billing", list(paperless_billing_rank.keys()))
    paperless_billing = paperless_billing_rank[paperless_billing]
    payment_method = st.selectbox("Payment Method", list(payment_method_rank.keys()))
    payment_method = payment_method_rank[payment_method]

    data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'SeniorCitizen': senior_citizen,
        'gender': gender,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service, 
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

# def preprocess_user_input(user_input, encoder):
#     # Perform necessary preprocessing on the user input
#     for column in user_input.columns:
#         if column in encoder:
#             user_input[column] = encoder[column].transform([user_input[column].iloc[0]])[0]
#         else:
#             print(column)
#     return user_input

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

    # Predict customer churn
    predict_button = st.button("Predict Churn")
    if predict_button:
        prediction = predict_churn(model, data)
        if prediction == 0:
            st.success("The customer is predicted to stay.")
        else:
            st.error("The customer is predicted to churn.")

if __name__ == "__main__":
    run()
