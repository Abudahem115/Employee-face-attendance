from flask import Flask, render_template, Response, jsonify, make_response, redirect, url_for
from modules.camera import VideoCamera
from modules import db_manager
import csv
import io

app = Flask(__name__)
app.secret_key = 'super_secret_key'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin')
def admin():
    users = db_manager.get_users_list()
    stats = db_manager.get_stats()
    return render_template('admin.html', users=users, stats=stats)

# --- تصحيح اسم الدالة هنا ليكون delete_user ---
@app.route('/delete_user/<int:user_id>')
def delete_user(user_id):
    if db_manager.delete_user(user_id):
        return redirect(url_for('admin'))
    else:
        return "Error deleting user"

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_attendance')
def get_attendance():
    data = db_manager.get_recent_attendance()
    return jsonify(data)

@app.route('/download_excel')
def download_excel():
    rows = db_manager.get_attendance_report()
    si = io.StringIO()
    si.write('\ufeff') 
    cw = csv.writer(si)
    cw.writerow(['Name', 'Timestamp']) 
    cw.writerows(rows)
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=attendance_report.csv"
    output.headers["Content-type"] = "text/csv; charset=utf-8-sig"
    return output

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)