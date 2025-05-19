import streamlit as st
import numpy as np
import pickle
from sklearn.datasets import load_iris

# Load model and data
iris = load_iris()
target_names = iris.target_names
model = pickle.load(open("model.pkl", "rb"))

# Image mapping (ensure these images are in the same folder or provide correct path)
flower_images = {
    "setosa": "setosa.jpg",
    "versicolor": "versicolor.jpg",
    "virginica": "virginica.jpg"
}

st.title("Iris Flower Classification")

st.write("Enter the features to predict the type of Iris flower:")

# Collect input features from the user
sepal_length = st.number_input("Sepal length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal width (cm)", min_value=0.0, step=0.1)

if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    prs = target_names[prediction][0]
    st.success(f"This flower is: {prs}")

    image_file = flower_images.get(prs)
    if image_file:
        st.image(image_file, caption=prs, use_column_width=True)
