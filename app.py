"""
app.py  ─  Task 9: Streamlit User Interface
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import joblib
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Evaluator",
    page_icon="🎓",
    layout="centered",
)

# ── Load model + scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

model, scaler = load_artifacts()

# ── Evaluation function (same as predict.py) ──────────────────────────────────
def evaluate_student(attendance, assignment, quiz, mid, study_hours):
    features = np.array([[attendance, assignment, quiz, mid, study_hours]])
    features_scaled = scaler.transform(features)
    pred_class = int(model.predict(features_scaled)[0])
    pred_proba = model.predict_proba(features_scaled)[0]
    confidence = float(pred_proba[pred_class]) * 100
    label = "Pass" if pred_class == 1 else "Fail"

    if pred_class == 1:
        if confidence >= 85:
            interp = "🌟 Excellent! Strong performance expected."
        elif confidence >= 70:
            interp = "✅ Good standing. Maintain your effort."
        else:
            interp = "⚠️ Borderline pass. Focus on weak areas."
    else:
        if confidence >= 85:
            interp = "❌ High risk of failure. Immediate improvement needed."
        elif confidence >= 70:
            interp = "⚠️ Likely to fail. Increase study hours & attendance."
        else:
            interp = "🔶 Borderline fail. Small improvements could change outcome."

    return {"result": pred_class, "label": label,
            "confidence": round(confidence, 1), "interpretation": interp}

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎓 Student Performance Evaluator")
st.markdown("**Powered by an Artificial Neural Network (ANN)**")
st.markdown(
    "Enter a student's academic data below and the ANN will predict "
    "whether they are likely to **Pass** or **Fail**."
)
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("📝 Student Input")

col1, col2 = st.columns(2)

with col1:
    attendance  = st.slider("Attendance (%)",    0,  100, 75)
    assignment  = st.slider("Assignment Score",   0,  100, 70)
    quiz        = st.slider("Quiz Score",         0,  100, 65)

with col2:
    mid         = st.slider("Mid-term Score",     0,  100, 55)
    study_hours = st.slider("Study Hours / week", 1,   15,  8)

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Evaluate Student", use_container_width=True, type="primary"):
    result = evaluate_student(attendance, assignment, quiz, mid, study_hours)

    # Result card
    if result["result"] == 1:
        st.success(f"## ✅ Prediction: **PASS**")
    else:
        st.error(f"## ❌ Prediction: **FAIL**")

    col_a, col_b = st.columns(2)
    col_a.metric("Confidence", f"{result['confidence']}%")
    col_b.metric("Result Code", result["result"])

    st.info(result["interpretation"])

    # Confidence bar
    st.markdown("**Model Confidence**")
    fail_prob = model.predict_proba(
        scaler.transform([[attendance, assignment, quiz, mid, study_hours]])
    )[0]
    st.markdown(f"Pass probability: **{fail_prob[1]*100:.1f}%**")
    st.progress(float(fail_prob[1]))

    st.divider()
    st.markdown("### 📋 Input Summary")
    st.table({
        "Feature":       ["Attendance", "Assignment", "Quiz", "Mid-term", "Study Hours"],
        "Entered Value": [attendance, assignment, quiz, mid, study_hours],
    })

# ── About section ─────────────────────────────────────────────────────────────
with st.expander("ℹ️ About this model"):
    st.markdown("""
    **Model:** MLPClassifier (Multi-Layer Perceptron) from scikit-learn  
    **Architecture:** 5 inputs → 64 neurons → 32 neurons → 1 output  
    **Trained on:** 480 student records (80% of 600-sample dataset)  
    **Activation:** ReLU (hidden), Softmax (output)  
    **Preprocessing:** StandardScaler (zero mean, unit variance)

    **Limitations:**
    - Trained on synthetic data — real-world accuracy may vary
    - Does not consider socioeconomic or psychological factors
    - Binary output only (Pass / Fail) — no nuanced performance levels
    - Small dataset may limit generalization
    """)
