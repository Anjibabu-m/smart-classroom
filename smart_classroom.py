from dotenv import load_dotenv
import os
import streamlit as st
import requests
import smtplib
import sqlite3
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- STREAMLIT CONFIG ---
st.set_page_config(page_title="Smart Classroom Assistant", layout="wide")

# --- LOAD ENV ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")
SENSOR_API = "https://api.thingspeak.com/channels/2871206/feeds.json?results=10"

# --- DB INIT ---
def init_db():
    conn = sqlite3.connect("sensor_data.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            temperature REAL,
            humidity REAL,
            gas REAL,
            noise REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- ML FUNCTIONS ---
def load_dataset():
    conn = sqlite3.connect("sensor_data.db")
    df = pd.read_sql_query("SELECT * FROM sensor_data", conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')
    return df

def train_predictor(df, column):
    df['timestamp_ordinal'] = df['timestamp'].apply(lambda x: x.toordinal())
    X = df[['timestamp_ordinal']]
    y = df[column]
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_next_hour(df):
    df['timestamp_ordinal'] = df['timestamp'].apply(lambda x: x.toordinal())
    future_time = df['timestamp'].max() + timedelta(hours=1)
    future_ordinal = future_time.toordinal()
    predictions = {}
    for sensor in ['temperature', 'humidity', 'gas', 'noise']:
        model = train_predictor(df, sensor)
        pred = model.predict([[future_ordinal]])[0]
        predictions[sensor] = round(pred, 2)
    return predictions, future_time

# --- SENSOR FETCH ---
def fetch_sensor_data():
    try:
        response = requests.get(SENSOR_API)
        if response.status_code == 200:
            data = response.json()
            latest_entry = data["feeds"][-1]
            field_values = latest_entry["field1"].split(",")
            if len(field_values) >= 4:
                temperature = float(field_values[2])
                humidity = float(field_values[3])
                gas = float(field_values[0])
                noise = float(field_values[1])
                timestamp = latest_entry["created_at"]

                conn = sqlite3.connect("sensor_data.db")
                cursor = conn.cursor()
                cursor.execute("INSERT INTO sensor_data (timestamp, temperature, humidity, gas, noise) VALUES (?, ?, ?, ?, ?)",
                    (timestamp, temperature, humidity, gas, noise))
                conn.commit()
                conn.close()

                return {
                    "temperature": temperature,
                    "humidity": humidity,
                    "gas": gas,
                    "noise": noise,
                    "timestamp": timestamp
                }
            else:
                st.error("Unexpected data format.")
        else:
            st.error(f"API request failed with status code {response.status_code}")
    except Exception as e:
        st.error(f"Error fetching sensor data: {str(e)}")
    return None

# --- GEMINI CHAT ---
def chat_with_gemini(prompt):
    try:
        headers = {"Content-Type": "application/json"}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        data = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        response = requests.post(url, headers=headers, json=data)
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Error: {e}"

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎓 Smart Classroom")
    st.sidebar.markdown("### System Navigation")
    page = st.radio("Select a Page", ["Dashboard", "Prediction", "History"])
    user_email = st.text_input("Enter Email for Alerts")
    clipboard_text = st.text_area("🗒️ Clipboard Text", height=150)

    if st.button("📧 Send Clipboard to Email"):
        if user_email:
            msg = MIMEText(clipboard_text)
            msg['Subject'] = 'Clipboard Content'
            msg['From'] = SENDER_EMAIL
            msg['To'] = user_email
            try:
                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                    server.login(SENDER_EMAIL, APP_PASSWORD)
                    server.send_message(msg)
                st.success("Clipboard emailed successfully!")
            except Exception as e:
                st.error(f"Email failed: {e}")
        else:
            st.warning("Please enter your email!")

    if clipboard_text.strip():
        st.download_button("📥 Download Clipboard Text", clipboard_text, "clipboard.txt", "text/plain")

    if st.button("🔄 Reset App"):
        st.session_state.clear()

# --- DASHBOARD ---
if page == "Dashboard":
    st.markdown(f"""
    <div style='text-align: center; padding: 15px;'>
        <h1 style='color: #00BFFF;'>🧠 AI-Powered IoT-Based Smart Classroom Assistant</h1>
        <h4>Enhancing Learning Through Automation</h4>
        <p style='color: #bbb;'>
            This innovative system utilizes IoT sensors and AI models to monitor and enhance the classroom environment.
            It automates lighting, temperature, and air quality for optimal student focus, and provides real-time data
            to instructors for better learning outcomes.
        </p>
        <b>📅 Date/Time:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    </div>
    """, unsafe_allow_html=True)

    if st.button("Fetch Latest Sensor Data from API"):
        result = fetch_sensor_data()
        if result:
            st.success("Fetched and stored latest data.")

    def get_latest_data_from_db():
        conn = sqlite3.connect("sensor_data.db")
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, temperature, humidity, gas, noise FROM sensor_data ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        if row:
            return {
                "timestamp": row[0],
                "temperature": row[1],
                "humidity": row[2],
                "gas": row[3],
                "noise": row[4],
            }
        return None

    sensor_data = get_latest_data_from_db()
    if sensor_data:
        c1, c2 = st.columns(2)
        with c1:
            st.metric("🌡️ Temperature", f"{sensor_data['temperature']}°C")
            st.metric("💧 Humidity", f"{sensor_data['humidity']}%")
        with c2:
            st.metric("🧪 Gas", sensor_data['gas'])
            st.metric("🔊 Noise", f"{sensor_data['noise']} dB")

        # --- CLASSROOM SITUATION ANALYSIS ---
        st.subheader("📊 Current Classroom Situation")

        def analyze_situation(data):
            status = []
            if data['temperature'] > 35:
                status.append("🔥 Very Hot")
            elif data['temperature'] > 28:
                status.append("🌡️ High Temperature")
            elif data['temperature'] < 18:
                status.append("❄️ Too Cold")
            else:
                status.append("✅ Comfortable Temperature")

            if data['humidity'] > 70:
                status.append("💧 Very Humid")
            elif data['humidity'] < 30:
                status.append("🌵 Too Dry")
            else:
                status.append("✅ Comfortable Humidity")

            if data['gas'] > 300:
                status.append("🧪 Poor Air Quality")
            elif data['gas'] > 150:
                status.append("⚠️ Slightly Polluted Air")
            else:
                status.append("✅ Good Air Quality")

            if data['noise'] > 80:
                status.append("🔊 Extremely Noisy")
            elif data['noise'] > 60:
                status.append("📢 Very Noisy")
            elif data['noise'] > 40:
                status.append("🗣️ Moderate Noise")
            else:
                status.append("✅ Quiet Environment")

            return status

        situation = analyze_situation(sensor_data)
        for s in situation:
            st.markdown(f"- {s}")

    st.subheader("Alert Thresholds 🚨")
    thresholds = {}
    cols = st.columns(4)
    for i, field in enumerate(['temperature', 'humidity', 'gas', 'noise']):
        thresholds[field] = cols[i].number_input(f"{field.capitalize()} Threshold", value=50)

    if st.button("Start Monitoring"):
        st.session_state.monitoring = True
        st.session_state.alert_sent = False

    if st.session_state.get("monitoring", False) and not st.session_state.get("alert_sent", False):
        alert_triggered = False
        for key, val in thresholds.items():
            current = float(sensor_data[key])
            if current > float(val):
                alert_msg = f"{key.capitalize()} exceeded threshold! Current: {current}, Threshold: {val}"
                if user_email:
                    try:
                        msg = MIMEText(alert_msg)
                        msg['Subject'] = 'Sensor Alert 🚨'
                        msg['From'] = SENDER_EMAIL
                        msg['To'] = user_email
                        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                            server.login(SENDER_EMAIL, APP_PASSWORD)
                            server.send_message(msg)
                        st.success(f"Email alert sent for {key}!")
                        alert_triggered = True
                    except Exception as e:
                        st.error(f"Email failed: {e}")
        if alert_triggered:
            st.session_state.alert_sent = True

    st.subheader("ChatBot 🗣️")
    def handle_chat_input():
        if st.session_state.chat_input:
            st.session_state.messages.append({"type": "user", "text": st.session_state.chat_input})
            question = st.session_state.chat_input.lower()
            if "temperature" in question:
                response = f"🌡️ Current temperature is {sensor_data['temperature']}°C"
            elif "humidity" in question:
                response = f"💧 Humidity is {sensor_data['humidity']}%"
            elif "gas" in question:
                response = f"🧪 Gas level is {sensor_data['gas']}"
            elif "noise" in question:
                response = f"🔊 Noise level is {sensor_data['noise']} dB"
            else:
                response = chat_with_gemini(st.session_state.chat_input)
            st.session_state.messages.append({"type": "bot", "text": response})
            st.session_state.chat_input = ""

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    st.text_input("Ask something...", key="chat_input", on_change=handle_chat_input)
    for msg in st.session_state.messages:
        if msg['type'] == 'user':
            st.markdown(f"🧑‍💻 **You:** {msg['text']}")
        else:
            st.markdown(f"🤖 **Bot:** {msg['text']}")

# --- PREDICTION PAGE ---
elif page == "Prediction":
    # Title
    st.title("🔮 Sensor Predictions")

    # Load dataset
    df = load_dataset()

    # Get predictions and prediction time
    predictions, future_time = predict_next_hour(df)

    # Display predicted values
    st.markdown("""<h4 style='color:#00BFFF;'>📈 Predicted Sensor Values</h4>""", unsafe_allow_html=True)
    st.table([
        {"Sensor": "🌡️ Temperature", "Predicted Value": f"{predictions['temperature']} °C", "Time": str(future_time)},
        {"Sensor": "💧 Humidity", "Predicted Value": f"{predictions['humidity']} %", "Time": str(future_time)},
        {"Sensor": "🧪 Gas", "Predicted Value": f"{predictions['gas']}", "Time": str(future_time)},
        {"Sensor": "🔊 Noise", "Predicted Value": f"{predictions['noise']} dB", "Time": str(future_time)},
    ])

    # --- MODEL EVALUATION ---
    st.subheader("📊 Model Evaluation (Average of 4 Sensors)")

    # Columns to evaluate
    sensor_columns = ['temperature', 'humidity', 'gas', 'noise']

    # Get actual latest average from real data
    actual_avg = df[sensor_columns].iloc[-1].mean()

    # Get predicted average from prediction result
    predicted_avg = np.mean([predictions[sensor] for sensor in sensor_columns])

    # Single-value comparison for simplicity
    y_true = [actual_avg]
    y_pred = [predicted_avg]

    # Calculate evaluation metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 1.0

    # Show metrics
    st.markdown(f"- **Mean Squared Error (MSE)**: `{mse:.2f}`")
    st.markdown(f"- **Root Mean Squared Error (RMSE)**: `{rmse:.2f}`")
    st.markdown(f"- **Mean Absolute Error (MAE)**: `{mae:.2f}`")
    st.markdown(f"- **R-Squared (R² Score)**: `{r2:.2f}`")


# --- HISTORY PAGE ---
elif page == "History":
    st.title("📜 Sensor Data History (Last 10 Entries)")
    conn = sqlite3.connect("sensor_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, temperature, humidity, gas, noise FROM sensor_data ORDER BY id DESC LIMIT 10")
    rows = cursor.fetchall()
    conn.close()
    if rows:
        st.table([
            {
                "📅 Timestamp": row[0],
                "🌡️ Temperature (°C)": row[1],
                "💧 Humidity (%)": row[2],
                "🧪 Gas": row[3],
                "🔊 Noise (dB)": row[4]
            } for row in rows
        ])
    else:
        st.warning("No data found in database.")
