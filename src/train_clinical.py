import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Load dataset
data = pd.read_csv("D:\MINI_PROJECT\data\meningitis.csv")

# Features & target
X = data.drop(columns=["Diagnosis", "Patient_ID"])
y = data["Diagnosis"]

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

# Encode categorical features
X = pd.get_dummies(X)

# Save feature columns
pickle.dump(X.columns, open("D:\MINI_PROJECT\models/model_features.pkl", "wb"))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train models
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
xgb.fit(X_train, y_train)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# Save models
pickle.dump(rf, open("D:\MINI_PROJECT\models/random_forest.pkl", "wb"))
pickle.dump(xgb, open("D:\MINI_PROJECT\models/xgboost.pkl", "wb"))
pickle.dump(lr, open("D:\MINI_PROJECT\models/logistic.pkl", "wb"))
pickle.dump(le, open("D:\MINI_PROJECT\models/label_encoder.pkl", "wb"))

print(" Models trained and saved!")