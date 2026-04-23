"""
ANN-Based Student Performance Evaluator
Tasks: 1, 3, 4, 5, 6, 8
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 – Understand the Dataset
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TASK 1: Dataset Overview")
print("=" * 60)

df = pd.read_excel("dataset.xlsx")

print("\n📋 First 5 Rows:")
print(df.head().to_string(index=False))

print("\n📐 Shape (rows × columns):", df.shape)

print("\n🏷️  Column Names:", df.columns.tolist())

print("\n📊 Column Descriptions:")
descriptions = {
    "attendance":   "Attendance percentage (0–100%)",
    "assignment":   "Assignment score (0–100)",
    "quiz":         "Quiz score (0–100)",
    "mid":          "Mid-term exam score (0–100)",
    "study_hours":  "Weekly study hours (1–15)",
    "result":       "Pass (1) or Fail (0) — TARGET variable",
}
for col, desc in descriptions.items():
    print(f"  • {col:15s} → {desc}")

print("\n🎯 Input Features  (X):", [c for c in df.columns if c != "result"])
print("🎯 Target Variable (y): result  (0 = Fail, 1 = Pass)")

print("\n📌 Problem Type: CLASSIFICATION")
print("   Justification: 'result' is binary (0/1). We predict a discrete",
      "category, not a continuous number.")

print("\n📈 Class Distribution:")
print(df["result"].value_counts().rename({0: "Fail (0)", 1: "Pass (1)"}).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 – Data Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 3: Data Preprocessing")
print("=" * 60)

X = df.drop("result", axis=1)
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n  Training samples : {len(X_train)}")
print(f"  Testing  samples : {len(X_test)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\n✅ StandardScaler applied.")
print("""
   WHY SCALING IS REQUIRED IN ANN:
   ─────────────────────────────────
   ANN computes weighted sums during forward-pass. If features have
   very different magnitudes (e.g. attendance 0–100 vs study_hours 1–15),
   large-valued features dominate weight updates, making training slow
   or biased. StandardScaler transforms each feature to mean=0, std=1
   so every feature contributes equally, enabling faster and more stable
   gradient descent convergence.
""")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 4 – Build ANN Model
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TASK 4: Build ANN Model")
print("=" * 60)

print("""
   ANN CONCEPTS:
   ─────────────
   • Neurons         : Basic computing units that receive inputs, apply
                       a weighted sum + bias, then pass through an
                       activation function.
   • Activation Fns  : Non-linear transforms (ReLU, tanh, sigmoid) that
                       let the network learn complex patterns beyond
                       simple linear relationships.
   • Hidden Layers   : Intermediate layers between input and output.
                       They extract progressively abstract features.
                       More layers = more expressive model (but risk of
                       overfitting on small data).
   
   Architecture chosen:
   Input Layer  → 5 neurons (one per feature)
   Hidden Layer 1 → 64 neurons, ReLU
   Hidden Layer 2 → 32 neurons, ReLU
   Output Layer → 1 neuron (sigmoid → class 0 or 1)
""")

model = MLPClassifier(
    hidden_layer_sizes=(64, 32),   # Two hidden layers
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42,
    verbose=False,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,
)

# ─────────────────────────────────────────────────────────────────────────────
# TASK 5 – Train the Model
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TASK 5: Training the ANN")
print("=" * 60)

model.fit(X_train_scaled, y_train)

print(f"\n✅ Training complete!")
print(f"   Iterations ran    : {model.n_iter_}")
best_loss = model.best_loss_ if model.best_loss_ is not None else model.loss_
print(f"   Best loss score   : {best_loss:.6f}")
print(f"   Converged         : {model.n_iter_ < model.max_iter}")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 6 – Evaluate Model
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 6: Model Evaluation")
print("=" * 60)

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"\n🎯 Accuracy : {acc * 100:.2f}%")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Fail (0)", "Pass (1)"]))

cm = confusion_matrix(y_test, y_pred)
print("🔲 Confusion Matrix:")
print(cm)
print("""
   CONFUSION MATRIX INTERPRETATION:
   ──────────────────────────────────
   Rows = Actual class   │  Cols = Predicted class
   ─────────────────────────────────────────────────
   TN (top-left)  : Correctly predicted Fail
   FP (top-right) : Fail student predicted as Pass   ← dangerous error
   FN (bot-left)  : Pass student predicted as Fail
   TP (bot-right) : Correctly predicted Pass

   ACCURACY means: out of all predictions, what fraction was correct?
   Mistakes:
   • FP (False Positives) — model is over-optimistic about some students
   • FN (False Negatives) — model is too pessimistic about some students
""")

# Bonus: Confusion Matrix Heatmap
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Fail", "Pass"],
    yticklabels=["Fail", "Pass"],
    ax=axes[0]
)
axes[0].set_title("Confusion Matrix", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

axes[1].plot(model.loss_curve_, color="#2563eb", lw=2, label="Training Loss")
if hasattr(model, "validation_scores_") and model.validation_scores_:
    val_loss = [1 - s for s in model.validation_scores_]
    axes[1].plot(val_loss, color="#dc2626", lw=2, linestyle="--", label="Validation Loss")
axes[1].set_title("Training Loss Curve", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_report.png", dpi=150)
print("\n📸 Saved: training_report.png")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 8 – Save Model + Scaler
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 8: Saving Model & Scaler")
print("=" * 60)

joblib.dump(model,  "model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("""
✅ Saved: model.joblib   — the trained ANN weights & architecture
✅ Saved: scaler.joblib  — the fitted StandardScaler

   WHY SAVE BOTH?
   ──────────────
   The scaler was fitted on training data (it knows mean & std of each
   feature). At inference time, new inputs MUST be scaled using the SAME
   fitted scaler — NOT re-fitted on new data. If only the model is saved
   and scaler is re-fitted later, the input values will be on a different
   scale than what the model was trained on, producing garbage predictions.
""")

print("=" * 60)
print("✅ All tasks complete! Run predict.py or app.py next.")
print("=" * 60)
