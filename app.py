import streamlit as st
import numpy as np
import pickle

# Load the pre-trained model from the file
def load_model():
    try:
        with open('C:\\Users\\yogin\\Desktop\\MLOps\\model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Make sure 'model.pkl' is in the correct location.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {str(e)}")
        return None

model = load_model()  # Load model at the beginning

# Title of the application
st.title("Credit Risk Prediction Application")

# Brief description
st.write("""
This application helps predict whether a borrower will default on their credit obligations based on their financial details.
Enter the following details to get a prediction.
""")

# Collecting input data from the user
SeriousDlqin2yrs = st.selectbox("Person experienced 90 days past due delinquency or worse", [0, 1])
RevolvingUtilizationOfUnsecuredLines = st.slider("Revolving Utilization of Unsecured Lines (percentage)", 0.0, 1.0, step=0.01)
age = st.number_input("Age of borrower", min_value=18, max_value=100, value=30)
NumberOfTime30_59DaysPastDueNotWorse = st.number_input("Number of times 30-59 days past due but no worse", min_value=0, max_value=20, value=0)
DebtRatio = st.slider("Debt Ratio (monthly debt payments divided by monthly gross income)", 0.0, 1.0, step=0.01)
MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, value=5000.0)
NumberOfOpenCreditLinesAndLoans = st.number_input("Number of Open Credit Lines and Loans", min_value=0, max_value=20, value=5)
NumberOfTimes90DaysLate = st.number_input("Number of times borrower has been 90 days or more past due", min_value=0, max_value=20, value=0)
NumberRealEstateLoansOrLines = st.number_input("Number of real estate loans or lines", min_value=0, max_value=10, value=1)
NumberOfTime60_89DaysPastDueNotWorse = st.number_input("Number of times 60-89 days past due but no worse", min_value=0, max_value=20, value=0)
NumberOfDependents = st.number_input("Number of dependents", min_value=0, max_value=10, value=0)

# Button to trigger prediction
if st.button("Predict Credit Risk"):
    # Check if the model is loaded
    if model is not None:
        # Create a numpy array for the model to predict from
        input_data = np.array([[SeriousDlqin2yrs,
                                RevolvingUtilizationOfUnsecuredLines,
                                age,
                                NumberOfTime30_59DaysPastDueNotWorse,
                                DebtRatio,
                                MonthlyIncome,
                                NumberOfOpenCreditLinesAndLoans,
                                NumberOfTimes90DaysLate,
                                NumberRealEstateLoansOrLines,
                                NumberOfTime60_89DaysPastDueNotWorse,
                                NumberOfDependents]])

        # Make prediction
        prediction = model.predict(input_data)

        # Show the result
        if prediction[0] == 1:
            st.error("High Credit Risk: The borrower is likely to default.")
        else:
            st.success("Low Credit Risk: The borrower is not likely to default.")
    else:
        st.error("Model is not loaded. Please check your model file.")
