import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load processed data
df = pd.read_csv("processed_data.csv")
print("✅ Data Loaded:", df.shape)

# Drop non-numeric / ID / metadata columns
drop_cols = [col for col in ["image_id", "lesion_id", "dx_type"] if col in df.columns]
df = df.drop(columns=drop_cols)

# Features and target
X = df.drop(columns=["dx"])
y = df["dx"]

# Ensure all features are numeric
print("📊 Feature dtypes after cleaning:\n", X.dtypes)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- Random Forest ---------------- #
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("\n🌲 Random Forest Results:")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_preds))
print("Classification Report:\n", classification_report(y_test, rf_preds, zero_division=0))

# Save model
joblib.dump(rf_model, "random_forest_model.pkl")

# ---------------- Logistic Regression ---------------- #
log_reg = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    solver="lbfgs",
    multi_class="auto"
)
log_reg.fit(X_train, y_train)
log_preds = log_reg.predict(X_test)

print("\n📈 Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, log_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, log_preds))
print("Classification Report:\n", classification_report(y_test, log_preds, zero_division=0))

# Save model
joblib.dump(log_reg, "logistic_regression_model.pkl")
