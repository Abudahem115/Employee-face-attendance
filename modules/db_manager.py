import sqlite3
import os
import pickle
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
DB_PATH = os.path.join(BASE_DIR, 'database', 'attendance.db')

def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            encoding BLOB NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"[INFO] Database Structure Ready at: {DB_PATH}")

def add_user_with_encodings(name, encodings_list):

    conn = get_db_connection()
    cursor = conn.cursor()
    try:

        cursor.execute('INSERT INTO users (name) VALUES (?)', (name,))
        user_id = cursor.lastrowid
        
        for encoding in encodings_list:
            encoding_blob = pickle.dumps(encoding)
            cursor.execute('INSERT INTO faces (user_id, encoding) VALUES (?, ?)', (user_id, encoding_blob))
            
        conn.commit()
        print(f"[SUCCESS] User '{name}' added with {len(encodings_list)} face samples.")
        return user_id
    except Exception as e:
        print(f"[ERROR] Adding user failed: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()

def get_all_embeddings():

    conn = get_db_connection()
    cursor = conn.cursor()
    
    sql = '''
        SELECT u.id, u.name, f.encoding 
        FROM users u
        JOIN faces f ON u.id = f.user_id
    '''
    cursor.execute(sql)
    rows = cursor.fetchall()
    conn.close()
    
    embeddings_data = []
    for row in rows:
        data = {
            "id": row["id"],
            "name": row["name"],
            "encoding": pickle.loads(row["encoding"])
        }
        embeddings_data.append(data)
    
    return embeddings_data

def mark_attendance(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        cursor.execute('INSERT INTO attendance (user_id, timestamp) VALUES (?, ?)', (user_id, now))
        conn.commit()
        print(f"[LOG] Attendance: User {user_id} at {now}")
    except Exception as e:
        print(f"[ERROR] Mark attendance failed: {e}")
    finally:
        conn.close()
        
# ... (أضف هذا في نهاية ملف db_manager.py)

def get_recent_attendance():
    """جلب آخر 5 أشخاص حضروا للعرض في الموقع"""
    conn = get_db_connection()
    cursor = conn.cursor()
    # نربط جدول الحضور بجدول الأسماء
    query = '''
        SELECT users.name, attendance.timestamp 
        FROM attendance 
        JOIN users ON attendance.user_id = users.id 
        ORDER BY attendance.timestamp DESC 
        LIMIT 5
    '''
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    # تحويل البيانات لقائمة قواميس
    return [{"name": row["name"], "time": row["timestamp"]} for row in rows]

def get_attendance_report():
    """جلب السجل كاملاً لملف الإكسل"""
    conn = get_db_connection()
    cursor = conn.cursor()
    query = '''
        SELECT users.name, attendance.timestamp 
        FROM attendance 
        JOIN users ON attendance.user_id = users.id 
        ORDER BY attendance.timestamp DESC
    '''
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return rows

# ... (أضف هذا في نهاية ملف modules/db_manager.py)

def get_users_list():
    """جلب قائمة الموظفين لعرضها في لوحة التحكم"""
    conn = get_db_connection()
    cursor = conn.cursor()
    # نجلب الاسم، الـ ID، وتاريخ التسجيل، وعدد البصمات المحفوظة
    query = '''
        SELECT u.id, u.name, u.created_at, COUNT(f.id) as face_count 
        FROM users u 
        LEFT JOIN faces f ON u.id = f.user_id 
        GROUP BY u.id
    '''
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return rows

def delete_user(user_id):
    """حذف موظف وجميع بياناته (بصمات + سجل حضور)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # 1. حذف بصمات الوجه
        cursor.execute('DELETE FROM faces WHERE user_id = ?', (user_id,))
        # 2. حذف سجلات الحضور
        cursor.execute('DELETE FROM attendance WHERE user_id = ?', (user_id,))
        # 3. حذف المستخدم نفسه
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        print(f"[INFO] User {user_id} deleted successfully.")
        return True
    except Exception as e:
        print(f"[ERROR] Delete failed: {e}")
        return False
    finally:
        conn.close()

def get_stats():
    """إحصائيات سريعة للوحة التحكم"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # عدد الموظفين
    cursor.execute('SELECT COUNT(*) FROM users')
    user_count = cursor.fetchone()[0]
    
    # عدد حضور اليوم
    today = datetime.now().strftime("%Y-%m-%d")
    cursor.execute('SELECT COUNT(DISTINCT user_id) FROM attendance WHERE timestamp LIKE ?', (f'{today}%',))
    attendance_count = cursor.fetchone()[0]
    
    conn.close()
    return {"users": user_count, "attendance": attendance_count}