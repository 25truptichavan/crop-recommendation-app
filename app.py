
import streamlit as st
import pandas as pd
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import pickle

# -----------------------------
# Load Saved Model
# -----------------------------
with open("crop_model.pkl", "rb") as f:
    rf = pickle.load(f)

# -----------------------------
# Load HuggingFace Model
# -----------------------------
pipe = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=120
)

llm = HuggingFacePipeline(pipeline=pipe)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Crop Recommendation System")

st.title("🌾 Crop Recommendation System")
st.write("Enter soil details to predict suitable crop.")

# Input Fields
N = st.number_input("Nitrogen (N)", min_value=0.0)
P = st.number_input("Phosphorus (P)", min_value=0.0)
K = st.number_input("Potassium (K)", min_value=0.0)
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("pH Value")
rainfall = st.number_input("Rainfall (mm)")

# Predict
if st.button("Predict Crop"):

    new_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

    prediction = rf.predict(new_data)[0]

    st.success(f"🌱 Recommended Crop: {prediction}")

    # AI Explanation Prompt
    prompt = f"""
    You are an agriculture expert.

    Soil Details:
    Nitrogen: {N}
    Phosphorus: {P}
    Potassium: {K}
    Temperature: {temperature}
    Humidity: {humidity}
    pH: {ph}
    Rainfall: {rainfall}

    The recommended crop is {prediction}.

    Explain why this crop is suitable in simple terms.
    """

    explanation = llm.invoke(prompt)

    st.subheader("🧠 AI Explanation")
    st.write(explanation)
