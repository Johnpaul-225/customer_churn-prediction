import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_excel("customer_churn_large_dataset.xlsx")

# Drop unnecessary columns
data.drop(["CustomerID", "Name"], axis=1, inplace=True)

# Encode categorical columns
le = LabelEncoder()
data["Gender"] = le.fit_transform(data["Gender"])
data["Location"] = le.fit_transform(data["Location"])

# Features & target
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# Save model & scaler
joblib.dump(model, "customer_churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved")
