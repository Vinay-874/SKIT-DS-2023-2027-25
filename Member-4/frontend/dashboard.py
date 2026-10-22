
import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Solar Power Forecasting",
    page_icon="☀️",
    layout="wide",
)

st.title("☀️ Solar Power Forecasting Dashboard")
st.write("Enter weather and solar conditions to generate a forecast.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        ghi = st.number_input("GHI (W/m²)", min_value=0.0, value=500.0)
        dni = st.number_input("DNI (W/m²)", min_value=0.0, value=400.0)
        dhi = st.number_input("DHI (W/m²)", min_value=0.0, value=100.0)
        temperature = st.number_input("Temperature (°C)", value=30.0)

    with col2:
        humidity = st.number_input(
            "Humidity (%)", min_value=0.0, max_value=100.0, value=50.0
        )
        pressure = st.number_input(
            "Pressure (hPa)", min_value=0.1, value=1013.0
        )
        wind_speed = st.number_input(
            "Wind Speed (m/s)", min_value=0.0, value=5.0
        )
        solar_zenith_angle = st.number_input(
            "Solar Zenith Angle (°)",
            min_value=0.0,
            max_value=180.0,
            value=45.0,
        )

    submitted = st.form_submit_button("Generate Forecast")

if submitted:
    payload = {
        "ghi": ghi,
        "dni": dni,
        "dhi": dhi,
        "temperature": temperature,
        "humidity": humidity,
        "pressure": pressure,
        "wind_speed": wind_speed,
        "solar_zenith_angle": solar_zenith_angle,
    }

    try:
        with st.spinner("Generating prediction..."):
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30,
            )

        if response.ok:
            result = response.json()
            prediction = result.get("prediction")

            if prediction is not None:
                st.success("Prediction generated successfully")
                st.metric("Predicted Solar Irradiance", f"{prediction:.2f} W/m²")

                chart_data = pd.DataFrame({
                    "Measurement": ["Predicted irradiance"],
                    "W/m²": [prediction],
                })
                st.bar_chart(chart_data, x="Measurement", y="W/m²")
            else:
                st.error("The API did not return a prediction.")
        else:
            st.error(f"API error ({response.status_code}): {response.text}")

    except requests.RequestException:
        st.error(
            "Cannot connect to the FastAPI backend. "
            "Make sure the backend server is running."
        )
