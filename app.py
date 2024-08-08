import streamlit as st
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

# Define the input form
def input_features():
    st.title("House Price Prediction")

    # Create input fields for each feature
    area = st.number_input("Area (in sq. ft.)", min_value=0.0, step=0.1)
    bedrooms = st.number_input("Number of Bedrooms", min_value=0, step=1)
    bathrooms = st.number_input("Number of Bathrooms", min_value=0, step=1)
    stories = st.number_input("Number of Stories", min_value=0, step=1)
    mainroad = st.selectbox("On Main Road", ["Yes", "No"])
    guestroom = st.selectbox("Guestroom Available", ["Yes", "No"])
    basement = st.selectbox("Basement", ["Yes", "No"])
    hotwaterheating = st.selectbox("Hot Water Heating", ["Yes", "No"])
    airconditioning = st.selectbox("Air Conditioning", ["Yes", "No"])
    parking = st.number_input("Parking Spaces", min_value=0, step=1)
    prefarea = st.selectbox("Preferred Area", ["Yes", "No"])
    furnishingstatus = st.selectbox("Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])

    # Convert categorical inputs to numerical values if needed
    mainroad = 1 if mainroad == "Yes" else 0
    guestroom = 1 if guestroom == "Yes" else 0
    basement = 1 if basement == "Yes" else 0
    hotwaterheating = 1 if hotwaterheating == "Yes" else 0
    airconditioning = 1 if airconditioning == "Yes" else 0
    prefarea = 1 if prefarea == "Yes" else 0
    # Convert furnishing status to a numerical value
    furnishingstatus = {"Furnished": 2, "Semi-Furnished": 1, "Unfurnished": 0}.get(furnishingstatus, 0)

    # Create a numpy array with the input features
    features = np.array([[
        area,
        bedrooms,
        bathrooms,
        stories,
        mainroad,
        guestroom,
        basement,
        hotwaterheating,
        airconditioning,
        parking,
        prefarea,
        furnishingstatus
    ]])

    return features

def format_price(price):
    """Convert the price into a readable format in lakhs and crores."""
    if price >= 1e7:
        return f"{price / 1e7:.2f} Crores"
    elif price >= 1e5:
        return f"{price / 1e5:.2f} Lakhs"
    else:
        return f"{price:.2f} Rs"

def loadModel(features):
    # Load the model
    lr = joblib.load('linear_regression_model.pkl')

    # Use the model to make predictions
    predictions = lr.predict(features)

    return predictions[0]  # Return the prediction for the first sample

# Create the main app
def main():
    features = input_features()
    
    if st.button("Predict"):     
        prediction = loadModel(features)
        formatted_price = format_price(prediction)
        st.write(f"Predicted Price: {formatted_price}")

if __name__ == "__main__":
    main()