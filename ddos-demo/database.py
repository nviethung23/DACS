import sqlite3
import threading
import time

_DB_PATH = "ddos_logs.db"
_lock = threading.Lock()

def init_db():
    with _lock:
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ip TEXT NOT NULL,
            result TEXT NOT NULL,
            detection_time REAL,
            block_time REAL,
            status_code INTEGER,
            timestamp INTEGER
        )
        """)
        conn.commit()
        conn.close()

def insert_log(ip, result, detection_time, block_time, status_code, timestamp):
    with _lock:
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO logs (ip, result, detection_time, block_time, status_code, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (ip, result, detection_time, block_time, status_code, timestamp))
        conn.commit()
        conn.close()

def query_recent_logs(limit=100):
    with _lock:
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT ip, result, detection_time, block_time, status_code, timestamp
            FROM logs
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
    # Trả về list of dict
    return [
        {
            "ip": row[0],
            "result": row[1],
            "detection_time": row[2],
            "block_time": row[3],
            "status_code": row[4],
            "timestamp": row[5]
        } for row in rows
    ]

def query_count_blocked_last_n_seconds(n_seconds=10):
    now = int(time.time())
    cutoff = now - n_seconds
    with _lock:
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM logs
            WHERE status_code != 200 AND timestamp >= ?
        """, (cutoff,))
        count = cursor.fetchone()[0]
        conn.close()
    return count
