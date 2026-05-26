import streamlit as st
import pickle
import numpy as np

# loading model
model = pickle.load(
    open("models/dtr_model.pkl", "rb")
)

st.set_page_config(
    page_title="Car Price Prediction",
    layout="centered"
)

st.title("Car Price Prediction using Decision Tree Regressor")

st.write("Enter car details below")

symboling = st.slider(
    "Symboling",
    -3,
    3,
    0
)

fueltype = st.selectbox(
    "Fuel Type",
    ["gas", "diesel"]
)

aspiration = st.selectbox(
    "Aspiration",
    ["std", "turbo"]
)

doornumber = st.selectbox(
    "Doors",
    ["two", "four"]
)

horsepower = st.slider(
    "Horsepower",
    40,
    300,
    100
)

peakrpm = st.slider(
    "Peak RPM",
    4000,
    7000,
    5000
)

citympg = st.slider(
    "City MPG",
    10,
    60,
    25
)

highwaympg = st.slider(
    "Highway MPG",
    10,
    60,
    30
)

fueltype = 1 if fueltype == "gas" else 0

aspiration = 1 if aspiration == "turbo" else 0

doornumber = 1 if doornumber == "two" else 0

input_data = np.array([[
    symboling,
    fueltype,
    aspiration,
    doornumber,
    2,
    1,
    100,
    150,
    64,
    5200,
    4,
    horsepower,
    peakrpm,
    citympg,
    highwaympg,
    2,
    120,
    95,
    50,
    20,
    30,
    1500,
    1
]])

if st.button("Predict Price"):

    prediction = model.predict(
        input_data
    )

    st.success(
        f"Predicted Car Price: ₹ {prediction[0]:,.2f}"
    )