from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash, make_response
from datetime import datetime
from modules import db_manager
from modules.camera import VideoCamera # الكاميرا العادية
import cv2
import face_recognition
import numpy as np
import pickle
import time
import io
import csv

app = Flask(__name__)
app.secret_key = 'secr3t_k3y'

# --- كلاس كاميرا التسجيل (لإضافة موظف جديد) ---
class RegistrationCamera:
    def __init__(self, user_name):
        self.video = cv2.VideoCapture(0)
        self.user_name = user_name
        self.encodings = []
        self.max_samples = 20 # عدد الصور المطلوبة
        self.is_finished = False

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success: return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # الرسم والتوجيه
        color = (0, 165, 255) # برتقالي
        msg = "Looking for face..."

        if len(face_locations) == 1:
            if len(self.encodings) < self.max_samples:
                try:
                    encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                    self.encodings.append(encoding)
                    msg = f"Capturing: {len(self.encodings)}/{self.max_samples}"
                    color = (0, 255, 0)
                    time.sleep(0.1) # تأخير بسيط
                except:
                    pass
            else:
                msg = "Done! Saving..."
                self.is_finished = True
        
        # رسم المربع والنص
        if len(face_locations) > 0:
            top, right, bottom, left = face_locations[0]
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def save_data(self):
        if self.encodings:
            db_manager.add_user_with_encodings(self.user_name, self.encodings)
            return True
        return False

# متغير عالمي لتخزين جلسة التسجيل الحالية
registration_session = None

# --- الروابط (Routes) ---

@app.route('/')
def dashboard():
    selected_month = request.args.get('month')
    
    # 1. General Stats
    stats = db_manager.get_dashboard_stats(selected_month)
    
    # 2. Charts & Dropdown
    available_months = db_manager.get_available_months()
    chart_data = db_manager.get_monthly_stats()
    
    # 3. NEW: Who is present/absent today
    daily_status = db_manager.get_daily_status()
    
    return render_template('dashboard.html', 
                         stats=stats, 
                         months=available_months, 
                         chart_data=chart_data,
                         daily_status=daily_status) # Pass this to HTML

# NEW ROUTE: Download Detailed Monthly Report
@app.route('/download_detailed_csv')
def download_detailed_csv():
    month = request.args.get('month')
    if not month:
        month = datetime.now().strftime('%Y-%m')
        
    data = db_manager.get_detailed_monthly_report_data(month)
    
    # Create CSV
    si = io.StringIO()
    si.write('\ufeff') # BOM for Excel support (Arabic)
    cw = csv.writer(si)
    
    # Header
    cw.writerow(['Employee Name', 'Days Present', 'Days Absent', 'Total Days in Month'])
    
    # Rows
    for row in data:
        cw.writerow([row['name'], row['present'], row['absent'], row['total']])
        
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = f"attachment; filename=Detailed_Report_{month}.csv"
    output.headers["Content-type"] = "text/csv; charset=utf-8-sig"
    return output

@app.route('/employees')
def employees():
    users = db_manager.get_users_list()
    return render_template('employees.html', users=users)

@app.route('/edit_employee/<int:user_id>', methods=['POST'])
def edit_employee(user_id):
    new_name = request.form.get('name')
    if new_name:
        db_manager.update_user(user_id, new_name)
        flash('Employee updated successfully', 'success')
    return redirect(url_for('employees'))

@app.route('/delete_employee/<int:user_id>')
def delete_employee(user_id):
    db_manager.delete_user(user_id)
    flash('Employee deleted successfully', 'danger')
    return redirect(url_for('employees'))

# --- صفحة إضافة موظف ---
@app.route('/add_employee', methods=['GET', 'POST'])
def add_employee():
    global registration_session
    
    if request.method == 'POST':
        name = request.form.get('name')
        if name:
            registration_session = RegistrationCamera(name)
            return render_template('training.html', name=name)
            
    return render_template('add_employee.html')

@app.route('/training_feed')
def training_feed():
    global registration_session
    def gen(camera):
        while True:
            if camera.is_finished:
                break
            frame = camera.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
    return Response(gen(registration_session), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_training_status')
def check_training_status():
    global registration_session
    if registration_session and registration_session.is_finished:
        registration_session.save_data()
        registration_session = None # إنهاء الجلسة
        return jsonify({'status': 'finished'})
    return jsonify({'status': 'training'})

# --- الكاميرا الرئيسية (Dashboard) ---
# (نفس دالة video_feed القديمة، يمكن وضعها في صفحة مستقلة أو في الداشبورد)
# هنا سأفترض أنك تريدها في صفحة "Live Monitor" منفصلة أو جزء من الداشبورد
@app.route('/live_monitor')
def live_monitor():
    return render_template('monitor.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames(camera):
    while True:
        frame = camera.get_frame()
        if frame: yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)