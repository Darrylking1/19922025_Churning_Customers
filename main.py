import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

def run():
    st.set_page_config(
        page_title="Customer Churn Prediction",
        page_icon="🧊",
    )

    st.markdown('<style>body{background-color: Blue;}</style>', unsafe_allow_html=True)

    loaded_model = tf.keras.models.load_model('mlp_model.h5')

    scaler = pickle.load(open('scaler.pkl', "rb"))
    encoder = pickle.load(open('encoder.pkl', "rb"))

    # Streamlit App
    st.title("Churn Prediction App")
    st.write("\n Enter Profile data \n")

    # Collect user input for prediction
    st.header("Enter Customer Information:")

    tenure = st.number_input("Tenure", min_value=0.0, step=1.0, value=48.65)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=1.0, value=92.85)
    total_charges = st.number_input("Total Charges", min_value=0.0, step=1.0, value=77.15)
    senior_citizen = st.selectbox("Senior Citizen", ['No', 'Yes'])
    gender = st.selectbox("Gender", ['Female', 'Male'])
    partner = st.selectbox("Partner", ['No', 'Yes'])
    dependents = st.selectbox("Dependents", ['No', 'Yes'])
    multiple_lines = st.selectbox("Multiple Lines", ['No phone service', 'No', 'Yes'])
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox("Online Security", ['No', 'Yes', 'No internet service'])
    online_backup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
    device_protection = st.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])
    tech_support = st.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
    streaming_movies = st.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'])
    contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
    payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
                                                     'Credit card (automatic)'])

    # Make prediction
    non_numeric = pd.DataFrame({
        'SeniorCitizen': [{'No': 0, 'Yes': 1}.get(senior_citizen, None)],
        'gender': [gender],
        'Partner': [partner],
        'Dependents': [dependents],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
    })

    for i in non_numeric:
        if i == 'SeniorCitizen':
            pass
        else:
            non_numeric[i] = encoder[i].transform(non_numeric[i])

    numeric = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
    })

    scaled_numeric_data = scaler.transform(numeric)
    numeric = pd.DataFrame(scaled_numeric_data, columns=numeric.columns)

    data = pd.concat([numeric, non_numeric], axis=1)

    if st.button("Predict"):
        # Perform prediction
        prediction = loaded_model.predict(data)

        # Display the prediction
        st.subheader("Prediction:")
        if prediction[0] == 0:
            st.success("The profile is predicted to stay.")
        else:
            st.error("The profile is predicted to churn.")

# Run the app
if __name__ == '__main__':
    run()

