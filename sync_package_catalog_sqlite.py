import csv
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path


DB_PATH = Path("instance/clinic.db")
CSV_PATH = Path("uploads/accudetek_packages_scraped.csv")


def clean(value):
    text = str(value or "").strip()
    if text == "Hermatology":
        return "Hematology"
    if "B Vaccine" in text and any(ord(ch) > 127 for ch in text):
        return "Hepa B Vaccine"
    return text


def normalize(value):
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def unique(values):
    out = []
    seen = set()
    for value in values:
        value = clean(value)
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def infer_category(name):
    lower = name.lower()
    if any(term in lower for term in ["x-ray", "xray", "chest", "ultrasound", "duplex", "ecg", "echo"]):
        return "Imaging/Cardiology"
    if "drug" in lower or lower.strip() in {"met", "thc"}:
        return "Drug Testing"
    if any(term in lower for term in ["physical exam", "consultation", "medical exam"]):
        return "Medical Consultation"
    return "Laboratory"


def infer_requirements(name, linked_row=None):
    if linked_row:
        return json.loads(linked_row[2] or "[]"), json.loads(linked_row[3] or "[]")

    category = infer_category(name)
    lower = name.lower()
    roles = []
    equipment = []
    if category == "Laboratory":
        roles += ["Registered Medical Technologists", "Laboratory Technicians"]
        if "cbc" in lower or "platelet" in lower or "hematology" in lower:
            equipment.append("Automated Hematology Analyzer")
        if any(term in lower for term in ["glucose", "creatinine", "cholesterol", "lipid", "uric", "sgpt", "sgot", "bun", "alkaline"]):
            equipment.append("Automated Clinical Chemistry Analyzer")
        if any(term in lower for term in ["sodium", "potassium", "chloride", "electrolyte"]):
            equipment.append("Automated Electrolyte Analyzer")
        if any(term in lower for term in ["thyroid", "hormone", "hepatitis", "serology", "immunology"]):
            equipment.append("Automated Immunoassay Analyzer")
        if any(term in lower for term in ["urine", "urinalysis", "fecalysis"]):
            equipment.append("Clinical microscopy laboratory resources")
    elif category == "Imaging/Cardiology":
        if any(term in lower for term in ["ecg", "echo"]):
            roles += ["General Physicians", "Trained ECG Staff"]
            equipment.append("Electrocardiograph (ECG) Machine")
        elif "ultrasound" in lower or "duplex" in lower:
            roles += ["Radiologists", "Registered Radiologic Technologists"]
            equipment.append("Ultrasound Machine")
        else:
            roles += ["Registered Radiologic Technologists", "Radiologists"]
            equipment.append("X-ray System/Machine")
    elif category == "Drug Testing":
        roles += ["Registered Medical Technologists", "Drug Test Analysts"]
        equipment.append("Drug testing laboratory resources")
    else:
        roles += ["General Physicians"]

    if not roles:
        roles = ["General Physicians"]
    return unique(roles), unique(equipment)


def main():
    con = sqlite3.connect(str(DB_PATH))
    cur = con.cursor()

    cur.execute("PRAGMA table_info(appointment)")
    appointment_cols = {row[1] for row in cur.fetchall()}
    if "selected_packages" not in appointment_cols:
        cur.execute("ALTER TABLE appointment ADD COLUMN selected_packages TEXT NOT NULL DEFAULT '[]'")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS clinic_package (
            id INTEGER NOT NULL PRIMARY KEY,
            package_name VARCHAR(180) NOT NULL UNIQUE,
            price_php VARCHAR(40),
            source_page VARCHAR(80),
            is_active BOOLEAN NOT NULL DEFAULT 1,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS package_item (
            id INTEGER NOT NULL PRIMARY KEY,
            package_id INTEGER NOT NULL,
            service_id INTEGER,
            item_name VARCHAR(180) NOT NULL,
            item_order INTEGER NOT NULL DEFAULT 0,
            required_roles TEXT NOT NULL DEFAULT '[]',
            required_equipment TEXT NOT NULL DEFAULT '[]',
            FOREIGN KEY(package_id) REFERENCES clinic_package (id),
            FOREIGN KEY(service_id) REFERENCES clinic_service (id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS branch_package (
            id INTEGER NOT NULL PRIMARY KEY,
            branch_id INTEGER NOT NULL,
            package_id INTEGER NOT NULL,
            custom_price_php VARCHAR(40),
            is_available BOOLEAN NOT NULL DEFAULT 1,
            branch_notes TEXT,
            updated_at DATETIME NOT NULL,
            UNIQUE (branch_id, package_id),
            FOREIGN KEY(branch_id) REFERENCES branch (id),
            FOREIGN KEY(package_id) REFERENCES clinic_package (id)
        )
    """)
    for statement in [
        "CREATE INDEX IF NOT EXISTS idx_clinic_package_name ON clinic_package(package_name)",
        "CREATE INDEX IF NOT EXISTS idx_package_item_package_id ON package_item(package_id)",
        "CREATE INDEX IF NOT EXISTS idx_package_item_service_id ON package_item(service_id)",
        "CREATE INDEX IF NOT EXISTS idx_branch_package_branch_id ON branch_package(branch_id)",
        "CREATE INDEX IF NOT EXISTS idx_branch_package_package_id ON branch_package(package_id)",
    ]:
        cur.execute(statement)

    cur.execute("SELECT id, service_name, required_roles, required_equipment FROM clinic_service")
    services = cur.fetchall()

    def linked_service(name):
        needle = normalize(name)
        if not needle:
            return None
        for row in services:
            if normalize(row[1]) == needle:
                return row
        for row in services:
            service_name = normalize(row[1])
            if needle in service_name or service_name in needle:
                return row
        return None

    cur.execute("SELECT package_name FROM clinic_package")
    existing = {clean(row[0]).lower() for row in cur.fetchall()}
    groups = {}
    if CSV_PATH.exists():
        with CSV_PATH.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                package_name = clean(row.get("package_name"))
                item = clean(row.get("included_service"))
                if package_name and item:
                    groups.setdefault(package_name, []).append(item)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    added_packages = 0
    added_items = 0
    for package_name, items in groups.items():
        if package_name.lower() in existing:
            continue
        cur.execute(
            "INSERT INTO clinic_package (package_name, price_php, source_page, is_active, created_at, updated_at) VALUES (?, ?, ?, 1, ?, ?)",
            (package_name, "", "packages", now, now),
        )
        package_id = cur.lastrowid
        for index, item in enumerate(unique(items), start=1):
            link = linked_service(item)
            roles, equipment = infer_requirements(item, link)
            cur.execute(
                "INSERT INTO package_item (package_id, service_id, item_name, item_order, required_roles, required_equipment) VALUES (?, ?, ?, ?, ?, ?)",
                (package_id, link[0] if link else None, item, index, json.dumps(roles), json.dumps(equipment)),
            )
            added_items += 1
        existing.add(package_name.lower())
        added_packages += 1

    cur.execute("SELECT id FROM branch")
    branch_ids = [row[0] for row in cur.fetchall()]
    cur.execute("SELECT id FROM clinic_package")
    package_ids = [row[0] for row in cur.fetchall()]
    branch_links_added = 0
    for branch_id in branch_ids:
        cur.execute("SELECT package_id FROM branch_package WHERE branch_id = ?", (branch_id,))
        linked = {row[0] for row in cur.fetchall()}
        for package_id in package_ids:
            if package_id in linked:
                continue
            cur.execute(
                "INSERT INTO branch_package (branch_id, package_id, custom_price_php, is_available, branch_notes, updated_at) VALUES (?, ?, ?, 1, ?, ?)",
                (branch_id, package_id, "", "", now),
            )
            branch_links_added += 1

    con.commit()
    cur.execute("SELECT COUNT(*) FROM clinic_package")
    package_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM package_item")
    item_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM branch_package")
    branch_package_count = cur.fetchone()[0]
    print("packages_added:", added_packages)
    print("package_items_added:", added_items)
    print("clinic_package_count:", package_count)
    print("package_item_count:", item_count)
    print("branch_package_links_added:", branch_links_added)
    print("branch_package_count:", branch_package_count)
    con.close()


if __name__ == "__main__":
    main()
