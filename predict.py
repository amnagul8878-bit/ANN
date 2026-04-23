"""
predict.py  ─  Task 7: Core Evaluation Function
Usage: python predict.py
"""

import numpy as np
import joblib

# Load saved artefacts
model  = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 7 – Core Evaluation Function
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_student(attendance, assignment, quiz, mid, study_hours):
    """
    Predict whether a student will Pass or Fail.

    Parameters
    ----------
    attendance   : int/float  – attendance percentage (0–100)
    assignment   : int/float  – assignment score (0–100)
    quiz         : int/float  – quiz score (0–100)
    mid          : int/float  – mid-term score (0–100)
    study_hours  : int/float  – weekly study hours (1–15)

    Returns
    -------
    dict with keys:
        result       – 0 (Fail) or 1 (Pass)
        label        – "Pass" or "Fail"
        confidence   – probability of the predicted class (%)
        interpretation – plain-English message
    """
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
            interp = "⚠️  Borderline pass. Focus on weak areas."
    else:
        if confidence >= 85:
            interp = "❌ High risk of failure. Immediate improvement needed."
        elif confidence >= 70:
            interp = "⚠️  Likely to fail. Increase study hours & attendance."
        else:
            interp = "🔶 Borderline fail. Small improvements could change outcome."

    return {
        "result":         pred_class,
        "label":          label,
        "confidence":     round(confidence, 1),
        "interpretation": interp,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI demo – evaluates a few sample students
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        dict(attendance=90, assignment=85, quiz=80, mid=75, study_hours=12),
        dict(attendance=40, assignment=35, quiz=30, mid=25, study_hours=2),
        dict(attendance=65, assignment=60, quiz=55, mid=50, study_hours=7),
    ]

    print("\n" + "=" * 60)
    print("  Student Evaluation Results")
    print("=" * 60)

    for i, s in enumerate(samples, 1):
        res = evaluate_student(**s)
        print(f"\n  Student #{i}")
        print(f"    Input  : att={s['attendance']}, asgn={s['assignment']}, "
              f"quiz={s['quiz']}, mid={s['mid']}, hrs={s['study_hours']}")
        print(f"    Result : {res['label']}  ({res['confidence']}% confidence)")
        print(f"    Note   : {res['interpretation']}")

    print("\n" + "=" * 60)
    print("  Interactive Mode")
    print("=" * 60)
    print("Enter student data (or Ctrl+C to quit):\n")

    try:
        while True:
            att  = float(input("  Attendance   (0–100) : "))
            asgn = float(input("  Assignment   (0–100) : "))
            quiz = float(input("  Quiz         (0–100) : "))
            mid  = float(input("  Mid-term     (0–100) : "))
            hrs  = float(input("  Study hours  (1–15)  : "))
            res  = evaluate_student(att, asgn, quiz, mid, hrs)
            print(f"\n  ➤ Prediction : {res['label']} ({res['confidence']}%)")
            print(f"  ➤ {res['interpretation']}\n")
    except KeyboardInterrupt:
        print("\n\nExiting.")
