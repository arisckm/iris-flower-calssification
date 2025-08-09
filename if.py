import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model and scaler
model = joblib.load("iris_model.pkl")
scaler = joblib.load("iris_scaler.pkl")

# Iris feature names
feature_names = ['sepal length (cm)', 'sepal width (cm)',
                 'petal length (cm)', 'petal width (cm)']

# Streamlit UI
st.title("🌸 Iris Flower Classification App")
st.write("Enter the flower's measurements to predict its species.")

# User inputs with sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Prepare input data
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=feature_names)

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)[0]
species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
predicted_species = species_map[prediction]

# Display result
st.subheader("Predicted Species:")
st.success(predicted_species)
