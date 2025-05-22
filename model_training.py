import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load your enriched dataset
df = pd.read_csv("Clean_WebServerLogs_Enriched (2).csv")

# Define target and features
target = "conversion"  # change if different
features = [col for col in df.columns if col not in [target, "timestamp", "session_id"]]  # adjust as needed

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Build pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

# Train and save model
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "logistic_model_pipeline.pkl")

print("✅ Model retrained and saved successfully.")
