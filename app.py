import streamlit as st
import pickle
import numpy as np

# loading saved models

pre_model = pickle.load(

    open(
        "models/prepruning_model.pkl",
        "rb"
    )
)

post_model = pickle.load(

    open(
        "models/postpruning_model.pkl",
        "rb"
    )
)

st.set_page_config(

    page_title="Car Price Prediction",

    layout="centered"
)

st.title(
    "Car Price Prediction using Decision Tree Regressor"
)

# selecting pruning method

model_choice = st.selectbox(

    "Choose Model",

    [
        "Pre Pruning",

        "Post Pruning"
    ]
)

# user inputs

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

enginesize = st.slider(
    "Engine Size",
    50,
    350,
    120
)

curbweight = st.slider(
    "Curb Weight",
    1000,
    5000,
    2500
)

wheelbase = st.slider(
    "Wheel Base",
    80,
    130,
    100
)

# encoding categorical inputs

fueltype = 1 if fueltype == "gas" else 0

aspiration = 1 if aspiration == "turbo" else 0

doornumber = 1 if doornumber == "two" else 0

# creating input array

input_data = np.array([[
    symboling,
    fueltype,
    aspiration,
    doornumber,
    horsepower,
    peakrpm,
    citympg,
    highwaympg,
    enginesize,
    curbweight,
    wheelbase
]])

if st.button("Predict Price"):

    # selecting model

    if model_choice == "Pre Pruning":

        prediction = pre_model.predict(
            input_data
        )

    else:

        prediction = post_model.predict(
            input_data
        )

    st.success(
        f"Predicted Car Price : ₹ {prediction[0]:,.2f}"
    )