import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(
    page_title="Road Accident Risk Predictor",
    page_icon="🚦",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #38bdf8;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 20px;
    margin-bottom: 30px;
}

.stButton>button {
    width: 100%;
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    background-color: #1d4ed8;
    color: white;
}

.result-box {
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    margin-top: 20px;
}

.safe {
    background-color: #14532d;
    color: #bbf7d0;
}

.careful {
    background-color: #78350f;
    color: #fde68a;
}

.danger {
    background-color: #7f1d1d;
    color: #fecaca;
}

.verydanger {
    background-color: #450a0a;
    color: #ff4d4d;
}

.suggestion-box {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    font-size: 18px;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load('models/accident_model.pkl')

# Load encoders
encoders = joblib.load('models/encoders.pkl')

# Header
st.markdown(
    '<div class="title">🚦 Smart Road Accident Risk Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Based Real-Time Accident Severity Analysis System</div>',
    unsafe_allow_html=True
)

# Layout
col1, col2 = st.columns(2)

with col1:

    st.subheader("📍 Accident Details")

    day = st.selectbox(
        "Day of Week",
        [
            'Monday',
            'Tuesday',
            'Wednesday',
            'Thursday',
            'Friday',
            'Saturday',
            'Sunday'
        ]
    )

    road_type = st.selectbox(
        "Road Type",
        [
            'Single carriageway',
            'Dual carriageway',
            'Roundabout',
            'One way street'
        ]
    )

    weather = st.selectbox(
        "Weather Condition",
        [
            'Clear',
            'Rainy',
            'Foggy'
        ]
    )

    light = st.selectbox(
        "Light Condition",
        [
            'Daylight',
            'Darkness'
        ]
    )

with col2:

    st.subheader("🚗 Vehicle Information")

    speed = st.slider(
        "Vehicle Speed",
        10,
        150,
        60
    )

    vehicles = st.slider(
        "Number of Vehicles",
        1,
        20,
        2
    )

    casualties = st.slider(
        "Number of Casualties",
        0,
        10,
        0
    )

# Analyze Button
if st.button("🔍 Analyze Accident Risk"):

    # Encode values
    try:
        day_encoded = encoders['Day_of_Week'].transform([day])[0]
    except:
        day_encoded = 0

    try:
        road_encoded = encoders['Road_Type'].transform([road_type])[0]
    except:
        road_encoded = 0

    # Input dataframe
    input_data = pd.DataFrame({
        'Road_Type': [road_encoded],
        'Speed_limit': [speed],
        'Number_of_Vehicles': [vehicles],
        'Number_of_Casualties': [casualties],
        'Day_of_Week': [day_encoded]
    })

    # ML prediction
    prediction = model.predict(input_data)[0]

    # Smart Risk Logic
    risk_score = 0

    if speed > 100:
        risk_score += 4
    elif speed > 70:
        risk_score += 2
    else:
        risk_score += 1

    if vehicles > 5:
        risk_score += 2

    if casualties > 2:
        risk_score += 3

    if weather in ['Rainy', 'Foggy']:
        risk_score += 2

    if light == 'Darkness':
        risk_score += 2

    # Risk Levels
    if risk_score <= 3:

        level = "✅ Very Safe"

        suggestion = """
        Road conditions appear safe.
        Continue following traffic rules
        and maintain safe driving habits.
        """

        css_class = "safe"

    elif risk_score <= 6:

        level = "⚠️ Be Careful"

        suggestion = """
        Moderate accident risk detected.
        Reduce distractions and maintain
        proper speed control.
        """

        css_class = "careful"

    elif risk_score <= 9:

        level = "🚨 Dangerous"

        suggestion = """
        High accident risk detected.
        Reduce speed immediately and
        maintain safe distance from vehicles.
        """

        css_class = "danger"

    else:

        level = "🔥 Very Dangerous"

        suggestion = """
        Critical accident risk detected.
        Avoid unnecessary travel and
        drive with maximum caution.
        """

        css_class = "verydanger"

    # Display Result
    st.markdown(
        f'<div class="result-box {css_class}">{level}</div>',
        unsafe_allow_html=True
    )

    # Suggestion
    st.markdown(
        f'<div class="suggestion-box">💡 <b>AI Safety Suggestion:</b><br><br>{suggestion}</div>',
        unsafe_allow_html=True
    )

    # Risk Meter
    st.subheader("📊 Risk Meter")

    st.progress(min(risk_score / 12, 1.0))

    # Technical prediction
    st.write(f"ML Severity Prediction: {prediction}")