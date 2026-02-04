import sqlite3
import csv

DB_PATH = "predictive_maintenance.db"
CSV_PATH = "predictive_maintenance.csv"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS maintenance (
        udi INTEGER PRIMARY KEY,
        product_id TEXT,
        type TEXT,
        air_temp REAL,
        process_temp REAL,
        rotational_speed INTEGER,
        torque REAL,
        tool_wear INTEGER,
        target INTEGER,
        failure_type TEXT
    )
    """)

    conn.commit()
    conn.close()


def import_csv():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    with open(CSV_PATH, newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",")
        # print("Detected columns:", reader.fieldnames)
        for row in reader:
            cur.execute("""
            INSERT OR REPLACE INTO maintenance VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(row["UDI"]),
                row["Product ID"],
                row["Type"],
                float(row["Air temperature [K]"]),
                float(row["Process temperature [K]"]),
                int(row["Rotational speed [rpm]"]),
                float(row["Torque [Nm]"]),
                int(row["Tool wear [min]"]),
                int(row["Target"]),
                row["Failure Type"]
            ))

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    import_csv()
    print("CSV imported successfully")