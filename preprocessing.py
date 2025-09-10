import pandas as pd
from sklearn.preprocessing import LabelEncoder

# File path
CSV_PATH = r"C:\Users\megav\Medinsights\dataset\HAM10000_metadata.csv"

# Load dataset
df = pd.read_csv(CSV_PATH)

# Drop missing values (if any)
df = df.dropna(subset=["sex", "age", "localization", "dx"])

# Encoders
sex_encoder = LabelEncoder()
loc_encoder = LabelEncoder()
dx_encoder = LabelEncoder()

# Encode categorical columns safely (no SettingWithCopyWarning)
df.loc[:, "sex"] = sex_encoder.fit_transform(df["sex"])
df.loc[:, "localization"] = loc_encoder.fit_transform(df["localization"])
df.loc[:, "dx"] = dx_encoder.fit_transform(df["dx"])

# Save encoders (optional: if you want later use for inference)
import joblib
joblib.dump(sex_encoder, "sex_encoder.pkl")
joblib.dump(loc_encoder, "loc_encoder.pkl")
joblib.dump(dx_encoder, "dx_encoder.pkl")

print("✅ Data Preprocessed:", df.shape)

# Save processed CSV
df.to_csv("processed_data.csv", index=False)
