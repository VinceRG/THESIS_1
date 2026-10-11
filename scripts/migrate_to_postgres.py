"""
migrate_to_postgres.py
Copies all data from local SQLite (instance/clinic.db)
to a target Neon PostgreSQL database.

Uses the DIRECT (non-pooler) Neon URL for both schema creation and data transfer.

Usage:
    python scripts/migrate_to_postgres.py "<direct_postgresql_url>"
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from sqlalchemy import create_engine, text, inspect, event

TABLE_ORDER = [
    'branch', 'user', 'staff_member', 'patient', 'patient_account',
    'medical_service', 'branch_service_setting', 'service_package',
    'service_package_item', 'branch_package_setting', 'consultation_record',
    'appointment', 'appointment_service_result',
    'chatbot_interaction', 'patient_chatbot_interaction', 'audit_log',
]


def normalize_url(url: str) -> str:
    url = url.replace('postgres://', 'postgresql://', 1)
    if url.startswith('postgresql://') and '+' not in url.split('://')[0]:
        url = url.replace('postgresql://', 'postgresql+psycopg2://', 1)
    return url


def migrate(postgres_url: str) -> bool:
    pg_url = normalize_url(postgres_url)
    host_display = pg_url.split('@')[-1] if '@' in pg_url else '...'
    print(f"\n[*] Target: {host_display}")

    # ── SQLite source ────────────────────────────────────────────────────────
    sqlite_path = os.path.join(ROOT, 'instance', 'clinic.db')
    if not os.path.exists(sqlite_path):
        print(f"[!] SQLite DB not found: {sqlite_path}")
        return False
    print(f"[*] Source: {sqlite_path}")
    sqlite_engine = create_engine(f'sqlite:///{sqlite_path}')

    # ── PostgreSQL target (autocommit for DDL) ───────────────────────────────
    pg_engine = create_engine(
        pg_url,
        connect_args={"connect_timeout": 30},
        isolation_level="AUTOCOMMIT"   # DDL needs autocommit on Neon
    )

    # Step 1 – Create schema using SQLAlchemy metadata directly on our engine
    print("[*] Creating PostgreSQL schema …")
    from app import db as flask_db
    # Import all models so metadata is populated
    import app as _app_module  # noqa: F401

    with pg_engine.connect() as conn:
        flask_db.metadata.create_all(bind=conn)

    # Step 2 – Verify schema
    with pg_engine.connect() as c:
        result = c.execute(
            text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
        )
        existing_pg_tables = [r[0] for r in result]

    print(f"[+] {len(existing_pg_tables)} tables in PostgreSQL: {', '.join(sorted(existing_pg_tables))}")
    if not existing_pg_tables:
        print("[!] No tables found after schema creation. Migration aborted.")
        return False

    # Step 3 – Migrate data (use transaction mode for inserts)
    pg_data_engine = create_engine(pg_url, connect_args={"connect_timeout": 30})
    sqlite_tables = inspect(sqlite_engine).get_table_names()
    ordered = [t for t in TABLE_ORDER if t in sqlite_tables]
    remaining = [t for t in sqlite_tables if t not in ordered]
    all_tables = ordered + remaining

    total_migrated = 0

    with sqlite_engine.connect() as src, pg_data_engine.connect() as dst:
        for table in all_tables:
            if table not in existing_pg_tables:
                print(f"\n[!] '{table}' not in PostgreSQL schema – skipping.")
                continue

            count = src.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar()
            print(f"\n[*] {table}  ({count} rows)")
            if count == 0:
                print("    (empty – skip)")
                continue

            sample = src.execute(text(f'SELECT * FROM "{table}" LIMIT 1'))
            columns = list(sample.keys())

            # Detect boolean columns from SQLAlchemy model metadata
            bool_columns = set()
            from sqlalchemy import Boolean
            if table in flask_db.metadata.tables:
                for col in flask_db.metadata.tables[table].columns:
                    if isinstance(col.type, Boolean):
                        bool_columns.add(col.name)

            dst.execute(text(f'TRUNCATE TABLE "{table}" CASCADE'))
            dst.commit()

            col_list = ', '.join(f'"{c}"' for c in columns)
            placeholders = ', '.join(f':{c}' for c in columns)
            insert_sql = text(f'INSERT INTO "{table}" ({col_list}) VALUES ({placeholders})')

            batch, offset = 500, 0
            while offset < count:
                rows = src.execute(
                    text(f'SELECT * FROM "{table}" LIMIT {batch} OFFSET {offset}')
                ).fetchall()
                if not rows:
                    break
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    # Coerce SQLite integer booleans (0/1) to Python bool for PostgreSQL
                    for bcol in bool_columns:
                        if bcol in row_dict and row_dict[bcol] is not None:
                            row_dict[bcol] = bool(row_dict[bcol])
                    dst.execute(insert_sql, row_dict)
                dst.commit()
                offset += len(rows)
                print(f"    {min(offset, count)}/{count}")


            # Reset primary-key sequence
            if 'id' in columns:
                try:
                    dst.execute(text(f"""
                        SELECT setval(
                            pg_get_serial_sequence('{table}', 'id'),
                            COALESCE((SELECT MAX(id) FROM "{table}"), 1),
                            true
                        )
                    """))
                    dst.commit()
                    print("    sequence reset ✓")
                except Exception as se:
                    dst.rollback()
                    print(f"    sequence note: {se}")

            total_migrated += count

    print(f"\n{'='*55}")
    print(f"  SUCCESS – {total_migrated:,} rows migrated to PostgreSQL!")
    print(f"{'='*55}\n")
    return True


if __name__ == '__main__':
    url = os.getenv('DATABASE_URL') or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not url:
        print(__doc__)
        sys.exit(1)
    sys.exit(0 if migrate(url) else 1)
