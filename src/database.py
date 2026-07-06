import sqlite3

def create_db():
    conn = sqlite3.connect("../database/meningitis.db")
    cursor = conn.cursor()

    cursor.execute("""
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
    conn.close()

def insert_data(pid, age, gender, prediction):
    conn = sqlite3.connect("../database/meningitis.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO patients (patient_id, age, gender, prediction)
    VALUES (?, ?, ?, ?)
    """, (pid, age, gender, prediction))

    conn.commit()
    conn.close()