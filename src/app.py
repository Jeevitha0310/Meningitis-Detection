import streamlit as st
import pandas as pd
import pickle
import sqlite3
import hashlib
import os

from PIL import Image
import torch
import timm
from torchvision import transforms

# ===========================
# PATH SETUP
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "database", "meningitis.db")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models")

# ===========================
# DATABASE CONNECTION
# ===========================
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

# ===========================
# USERS TABLE
# ===========================
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    email TEXT PRIMARY KEY,
    password TEXT
)
""")

# ===========================
# PATIENTS TABLE (YOUR DB)
# ===========================
c.execute("""
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT,
    age INTEGER,
    gender TEXT,
    prediction TEXT,
    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()

# ===========================
# PASSWORD HASH
# ===========================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ===========================
# LOAD MODELS
# ===========================
rf_model = pickle.load(open(os.path.join("D:\MINI_PROJECT\models", "random_forest.pkl"), "rb"))
le = pickle.load(open(os.path.join("D:\MINI_PROJECT\models", "label_encoder.pkl"), "rb"))
model_features = pickle.load(open(os.path.join("D:\MINI_PROJECT\models", "model_features.pkl"), "rb"))

# ===========================
# IMAGE MODEL
# ===========================
@st.cache_resource
def load_image_model():
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(os.path.join("D:\MINI_PROJECT\models", "vit_model.pth"), map_location="cpu"))
    model.eval()
    return model

image_model = load_image_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ===========================
# SESSION STATE
# ===========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.email = ""

# ===========================
# LOGIN / REGISTER
# ===========================
def login_page():
    st.title("🔐 Login / Register")

    option = st.selectbox("Choose", ["Login", "Register"])
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button(option):

        if option == "Register":
            c.execute("SELECT * FROM users WHERE email=?", (email,))
            if c.fetchone():
                st.error("User already exists")
            else:
                c.execute("INSERT INTO users VALUES (?, ?)", (email, hash_password(password)))
                conn.commit()
                st.session_state.logged_in = True
                st.session_state.email = email
                st.rerun()

        else:
            c.execute("SELECT * FROM users WHERE email=? AND password=?",
                      (email, hash_password(password)))
            if c.fetchone():
                st.session_state.logged_in = True
                st.session_state.email = email
                st.rerun()
            else:
                st.error("Invalid credentials")

# ===========================
# MAIN APP
# ===========================
def main_app():
    st.title("🧠 Meningitis Detection System")
    st.sidebar.success(f"Logged in: {st.session_state.email}")

    menu = st.sidebar.selectbox("Menu", ["Prediction", "History", "Logout"])

    # ================= PREDICTION =================
    if menu == "Prediction":

        with st.form("form"):
            patient_id = st.text_input("Patient ID")
            age = st.number_input("Age", 0, 120)
            gender = st.selectbox("Gender", ["Male", "Female"])
            wbc = st.number_input("WBC Count")
            protein = st.number_input("Protein Level")
            glucose = st.number_input("Glucose Level")
            pathogen = st.selectbox("Pathogen Present", ["Yes", "No"])
            hemo = st.number_input("Hemoglobin")
            wbc_blood = st.number_input("WBC Blood Count")
            platelets = st.number_input("Platelets")
            crp = st.number_input("CRP Level")

            image_file = st.file_uploader("Upload MRI/CT Image", type=["jpg", "png"])
            submit = st.form_submit_button("Predict")

        if submit:

            # ---------- ML INPUT ----------
            ml_input = {
                "Age": age,
                "Gender": gender,
                "WBC Count": wbc,
                "Protein Level": protein,
                "Glucose Level": glucose,
                "Pathogen Present": pathogen,
                "Hemoglobin": hemo,
                "WBC Blood Count": wbc_blood,
                "Platelets": platelets,
                "CRP Level": crp
            }

            df = pd.DataFrame([ml_input])
            df = pd.get_dummies(df)
            df = df.reindex(columns=model_features, fill_value=0)

            clinical_pred = le.inverse_transform(rf_model.predict(df))[0]
            st.success(f"🧪 Clinical Prediction: {clinical_pred}")

            # ---------- IMAGE ----------
            image_pred = None

            if image_file:
                image = Image.open(image_file).convert("RGB")
                img = transform(image).unsqueeze(0)

                with torch.no_grad():
                    output = image_model(img)
                    pred = torch.argmax(output, 1).item()

                image_pred = "Bacterial" if pred == 0 else "Viral"

                st.image(image, width=200)
                st.info(f"🖼 Image Prediction: {image_pred}")

            final = clinical_pred

            if image_pred:
                final = clinical_pred if clinical_pred == image_pred else clinical_pred

            st.success(f"🎯 Final Prediction: {final}")

            # ---------- SAVE TO DATABASE ----------
            c.execute("""
                INSERT INTO patients (patient_id, age, gender, prediction)
                VALUES (?, ?, ?, ?)
            """, (patient_id, age, gender, final))

            conn.commit()

    # ================= HISTORY =================
    elif menu == "History":
        st.subheader("📜 Patient History")

        c.execute("""
            SELECT id, patient_id, age, gender, prediction, date
            FROM patients
            ORDER BY id DESC
        """)

        data = c.fetchall()

        if data:
            df = pd.DataFrame(
                data,
                columns=["ID", "Patient ID", "Age", "Gender", "Prediction", "Date"]
            )

            st.dataframe(df)

            # DELETE ALL
            if st.button("🗑 Delete All History"):
                c.execute("DELETE FROM patients")
                conn.commit()
                st.success("All history deleted")
                st.rerun()

            # DELETE SINGLE
            delete_id = st.number_input("Enter ID to delete", min_value=0, step=1)

            if st.button("Delete Record"):
                c.execute("DELETE FROM patients WHERE id=?", (delete_id,))
                conn.commit()
                st.success("Record deleted")
                st.rerun()

        else:
            st.info("No history found")

    # ================= LOGOUT =================
    elif menu == "Logout":
        st.session_state.logged_in = False
        st.session_state.email = ""
        st.rerun()

# ===========================
# RUN APP
# ===========================
if not st.session_state.logged_in:
    login_page()
else:
    main_app()