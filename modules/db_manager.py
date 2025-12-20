import sqlite3
import os
import pickle
from datetime import datetime
import csv # مهم جداً للأرشفة
import calendar

# إعداد المسارات
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
    cursor.execute('''
        SELECT u.id, u.name, f.encoding 
        FROM users u
        JOIN faces f ON u.id = f.user_id
    ''')
    rows = cursor.fetchall()
    conn.close()
    embeddings_data = []
    for row in rows:
        embeddings_data.append({
            "id": row["id"],
            "name": row["name"],
            "encoding": pickle.loads(row["encoding"])
        })
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

def get_recent_attendance():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT users.name, attendance.timestamp 
        FROM attendance 
        JOIN users ON attendance.user_id = users.id 
        ORDER BY attendance.timestamp DESC 
        LIMIT 5
    ''')
    rows = cursor.fetchall()
    conn.close()
    return [{"name": row["name"], "time": row["timestamp"]} for row in rows]

def get_attendance_report():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT users.name, attendance.timestamp 
        FROM attendance 
        JOIN users ON attendance.user_id = users.id 
        ORDER BY attendance.timestamp DESC
    ''')
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_users_list():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT u.id, u.name, u.created_at, COUNT(f.id) as face_count 
        FROM users u 
        LEFT JOIN faces f ON u.id = f.user_id 
        GROUP BY u.id
    ''')
    rows = cursor.fetchall()
    conn.close()
    return rows

def delete_user(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('DELETE FROM faces WHERE user_id = ?', (user_id,))
        cursor.execute('DELETE FROM attendance WHERE user_id = ?', (user_id,))
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()

def get_stats():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM users')
    user_count = cursor.fetchone()[0]
    today = datetime.now().strftime("%Y-%m-%d")
    cursor.execute('SELECT COUNT(DISTINCT user_id) FROM attendance WHERE timestamp LIKE ?', (f'{today}%',))
    attendance_count = cursor.fetchone()[0]
    conn.close()
    return {"users": user_count, "attendance": attendance_count}

# --- الدوال الجديدة (سبب الخطأ) ---

def get_monthly_stats():
    """جلب إحصائيات آخر 6 شهور"""
    conn = get_db_connection()
    cursor = conn.cursor()
    # تجميع حسب السنة والشهر
    query = '''
        SELECT strftime('%Y-%m', timestamp) as month, COUNT(DISTINCT user_id) as count 
        FROM attendance 
        GROUP BY month 
        ORDER BY month DESC 
        LIMIT 6
    '''
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    
    labels = []
    data = []
    for row in reversed(rows):
        labels.append(row['month'])
        data.append(row['count'])
    return {"labels": labels, "data": data}

def get_available_months():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT strftime('%Y-%m', timestamp) as month FROM attendance ORDER BY month DESC")
    rows = cursor.fetchall()
    conn.close()
    return [row['month'] for row in rows]

def get_attendance_by_month(month_str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT users.name, attendance.timestamp 
        FROM attendance 
        JOIN users ON attendance.user_id = users.id 
        WHERE strftime('%Y-%m', attendance.timestamp) = ?
        ORDER BY attendance.timestamp DESC
    ''', (month_str,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def archive_and_clear():
    archive_dir = os.path.join(BASE_DIR, 'archives')
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)
        
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT users.name, attendance.timestamp 
        FROM attendance 
        JOIN users ON attendance.user_id = users.id 
    ''')
    rows = cursor.fetchall()
    
    if not rows:
        conn.close()
        return None 

    filename = f"archive_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    filepath = os.path.join(archive_dir, filename)
    
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Timestamp'])
        for row in rows:
            writer.writerow([row['name'], row['timestamp']])
            
    cursor.execute('DELETE FROM attendance')
    conn.commit()
    conn.close()
    return filename

# دالة جديدة: تعديل بيانات الموظف
def update_user(user_id, new_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('UPDATE users SET name = ? WHERE id = ?', (new_name, user_id))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error updating user: {e}")
        return False
    finally:
        conn.close()
        
def get_dashboard_stats(selected_month=None):
    """
    selected_month format: 'YYYY-MM'
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if not selected_month:
        selected_month = datetime.now().strftime('%Y-%m')

    # إحصائيات الشهر المحدد
    cursor.execute('''
        SELECT COUNT(DISTINCT user_id) 
        FROM attendance 
        WHERE strftime('%Y-%m', timestamp) = ?
    ''', (selected_month,))
    monthly_attendance = cursor.fetchone()[0]

    # إجمالي الموظفين
    cursor.execute('SELECT COUNT(*) FROM users')
    total_users = cursor.fetchone()[0]
    
    conn.close()
    return {
        "total_users": total_users,
        "monthly_attendance": monthly_attendance,
        "selected_month": selected_month
    }
    
def get_daily_status():
    """
    Returns two lists: Users present today, and Users absent today.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Get all users
    cursor.execute('SELECT id, name FROM users')
    all_users = cursor.fetchall()
    
    # Get users who attended today
    cursor.execute('''
        SELECT DISTINCT user_id 
        FROM attendance 
        WHERE timestamp LIKE ?
    ''', (f'{today}%',))
    present_ids = [row['user_id'] for row in cursor.fetchall()]
    
    conn.close()
    
    present_list = []
    absent_list = []
    
    for user in all_users:
        if user['id'] in present_ids:
            present_list.append(user)
        else:
            absent_list.append(user)
            
    return {"present": present_list, "absent": absent_list}

def get_detailed_monthly_report_data(month_str):
    """
    Calculates detailed stats for CSV: Name, Days Present, Days Absent
    month_str format: 'YYYY-MM'
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Get total days passed in that month (to calculate absence)
    year, month = map(int, month_str.split('-'))
    _, num_days_in_month = calendar.monthrange(year, month)
    
    # If viewing current month, only count up to today
    now = datetime.now()
    if now.year == year and now.month == month:
        total_working_days = now.day
    else:
        total_working_days = num_days_in_month

    # 2. Get all users
    cursor.execute('SELECT id, name FROM users')
    users = cursor.fetchall()
    
    report_data = []
    
    for user in users:
        # Count distinct days attended in this specific month
        cursor.execute('''
            SELECT COUNT(DISTINCT date(timestamp)) 
            FROM attendance 
            WHERE user_id = ? AND strftime('%Y-%m', timestamp) = ?
        ''', (user['id'], month_str))
        
        days_present = cursor.fetchone()[0]
        days_absent = total_working_days - days_present
        
        # Prevent negative absence numbers (just in case)
        if days_absent < 0: days_absent = 0
        
        report_data.append({
            "name": user['name'],
            "present": days_present,
            "absent": days_absent,
            "total": total_working_days
        })
        
    conn.close()
    return report_data