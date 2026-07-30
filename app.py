import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

st.set_page_config(page_title="Mobile Addiction Level Predictor", page_icon="📱", layout="centered")

# ---------------------------------------------------------------
# Load artifacts (must be in the same folder as this file)
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("mobile_addiction_best_ann.keras")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    categorical_options = joblib.load("categorical_options.pkl")
    return model, scaler, label_encoders, feature_columns, categorical_options

model, scaler, label_encoders, feature_columns, categorical_options = load_artifacts()

st.title("📱 Mobile Addiction Level Predictor")
st.write(
    "Fill in the details below to predict a person's phone **Addiction Level** "
    "(scale roughly 0–10) using a trained neural network."
)

st.divider()

# ---------------------------------------------------------------
# Input form
# ---------------------------------------------------------------
with st.form("prediction_form"):
    st.subheader("Demographics")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=5, max_value=100, value=18)
        gender = st.selectbox("Gender", categorical_options["Gender"])
    with col2:
        purpose = st.selectbox("Primary Phone Usage Purpose", categorical_options["Phone_Usage_Purpose"])

    st.subheader("Usage Habits")
    col3, col4 = st.columns(2)
    with col3:
        daily_usage = st.number_input("Daily Usage Hours", min_value=0.0, max_value=24.0, value=4.0, step=0.1)
        weekend_usage = st.number_input("Weekend Usage Hours", min_value=0.0, max_value=24.0, value=5.0, step=0.1)
        screen_before_bed = st.number_input("Screen Time Before Bed (hrs)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        phone_checks = st.number_input("Phone Checks Per Day", min_value=0, max_value=500, value=80)
        apps_used = st.number_input("Apps Used Daily", min_value=0, max_value=100, value=10)
    with col4:
        time_social = st.number_input("Time on Social Media (hrs)", min_value=0.0, max_value=24.0, value=2.0, step=0.1)
        time_gaming = st.number_input("Time on Gaming (hrs)", min_value=0.0, max_value=24.0, value=1.0, step=0.1)
        time_education = st.number_input("Time on Education Apps (hrs)", min_value=0.0, max_value=24.0, value=1.0, step=0.1)

    st.subheader("Wellbeing & Lifestyle")
    col5, col6 = st.columns(2)
    with col5:
        sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.1)
        exercise_hours = st.number_input("Exercise Hours (daily)", min_value=0.0, max_value=10.0, value=0.5, step=0.1)
        intellectual_perf = st.slider("Intellectual Performance", 0, 100, 70)
    with col6:
        anxiety = st.slider("Anxiety Level", 0, 10, 3)
        depression = st.slider("Depression Level", 0, 10, 3)
        self_esteem = st.slider("Self Esteem", 0, 10, 6)

    col7, col8 = st.columns(2)
    with col7:
        social_interactions = st.slider("Social Interactions (daily count)", 0, 20, 5)
    with col8:
        family_comm = st.slider("Family Communication (score)", 0, 10, 5)

    submitted = st.form_submit_button("Predict Addiction Level", use_container_width=True)

# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
if submitted:
    raw_input = {
        "Age": age,
        "Gender": gender,
        "Daily_Usage_Hours": daily_usage,
        "Sleep_Hours": sleep_hours,
        "Interllectual_Performance": intellectual_perf,
        "Social_Interactions": social_interactions,
        "Exercise_Hours": exercise_hours,
        "Anxiety_Level": anxiety,
        "Depression_Level": depression,
        "Self_Esteem": self_esteem,
        "Screen_Time_Before_Bed": screen_before_bed,
        "Phone_Checks_Per_Day": phone_checks,
        "Apps_Used_Daily": apps_used,
        "Time_on_Social_Media": time_social,
        "Time_on_Gaming": time_gaming,
        "Time_on_Education": time_education,
        "Phone_Usage_Purpose": purpose,
        "Family_Communication": family_comm,
        "Weekend_Usage_Hours": weekend_usage,
    }

    # Encode categoricals safely (fallback to most frequent class if unseen)
    for col in ["Gender", "Phone_Usage_Purpose"]:
        enc = label_encoders[col]
        value = raw_input[col]
        if value in enc.classes_:
            raw_input[col] = enc.transform([value])[0]
        else:
            fallback = enc.classes_[0]
            raw_input[col] = enc.transform([fallback])[0]

    # Build feature vector in the exact training column order
    x = np.array([[raw_input[col] for col in feature_columns]])
    x_scaled = scaler.transform(x)

    prediction = model.predict(x_scaled, verbose=0)[0][0]
    prediction = float(np.clip(prediction, 0, 10))

    st.divider()
    st.subheader("Result")
    st.metric("Predicted Addiction Level", f"{prediction:.2f} / 10")

    if prediction >= 7:
        st.error("High addiction risk. Consider reducing screen time and daily phone checks.")
    elif prediction >= 4:
        st.warning("Moderate addiction level. Some usage patterns could be improved.")
    else:
        st.success("Low addiction level. Healthy usage pattern.")

st.divider()
st.caption("Model: Keras Sequential ANN (64→32→dropout→16→1) trained on the phone addiction dataset.")