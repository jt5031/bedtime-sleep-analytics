import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, encoders, and target encoder
@st.cache_resource
def load_artifacts():
    model = joblib.load('model.pkl')
    encoders = joblib.load('label_encoders.pkl')
    target_encoder = joblib.load('target_encoder.pkl')
    return model, encoders, target_encoder

model, encoders, target_encoder = load_artifacts()

st.title("💤 Bedtime Screentime & Sleep Debt Predictor")
st.write("Analyze how your pre-sleep habits, screentime, and lifestyle choices impact your sleep debt and fatigue.")

# User Inputs
age = st.slider("Age", 18, 70, 30)
gender = st.selectbox("Gender", encoders['gender'].classes_)
occupation = st.selectbox("Occupation Type", encoders['occupation_type'].classes_)
chronotype = st.selectbox("Chronotype", encoders['chronotype'].classes_)
bedtime_phone_minutes = st.slider("Bedtime Phone Minutes", 0, 180, 45)
primary_app = st.selectbox("Primary Bedtime App", encoders['primary_bedtime_app'].classes_)
screen_brightness = st.slider("Screen Brightness (%)", 10, 100, 50)
blue_light_active = st.selectbox("Blue Light Filter Active?", [0, 1])
caffeine = st.slider("Caffeine Post-5 PM (mg)", 0, 250, 20)
physical_activity = st.slider("Physical Activity (minutes)", 0, 120, 30)
sleep_latency = st.slider("Sleep Latency (minutes - time to fall asleep)", 5.0, 120.0, 30.0)
total_sleep_hours = st.slider("Total Sleep Hours", 3.0, 10.0, 7.0)
deep_sleep_pct = st.slider("Deep Sleep (%)", 10.0, 30.0, 20.0)
rem_sleep_pct = st.slider("REM Sleep (%)", 10.0, 30.0, 20.0)
morning_alarm_snoozes = st.slider("Morning Alarm Snoozes", 0, 7, 2)
next_day_fatigue = st.slider("Next Day Fatigue Score (1-10)", 1.0, 10.0, 3.0)

if st.button("Predict Sleep Debt Risk"):
    # Create input dataframe
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [encoders['gender'].transform([gender])[0]],
        'occupation_type': [encoders['occupation_type'].transform([occupation])[0]],
        'chronotype': [encoders['chronotype'].transform([chronotype])[0]],
        'bedtime_phone_minutes': [bedtime_phone_minutes],
        'primary_bedtime_app': [encoders['primary_bedtime_app'].transform([primary_app])[0]],
        'screen_brightness_pct': [screen_brightness],
        'blue_light_filter_active': [blue_light_active],
        'caffeine_post_5pm_mg': [caffeine],
        'physical_activity_min': [physical_activity],
        'sleep_latency_min': [sleep_latency],
        'total_sleep_hours': [total_sleep_hours],
        'deep_sleep_pct': [deep_sleep_pct],
        'rem_sleep_pct': [rem_sleep_pct],
        'morning_alarm_snoozes': [morning_alarm_snoozes],
        'next_day_fatigue_score': [next_day_fatigue]
    })
    
    # Predict numerical class and convert back to text label
    pred_idx = model.predict(input_data)[0]
    prediction = target_encoder.inverse_transform([pred_idx])[0]
    
    st.success(f"Predicted Sleep Debt Category: **{prediction}**")