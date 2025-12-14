import cv2
import face_recognition
import numpy as np
from modules import db_manager
import time
import os
from datetime import datetime
from scipy.spatial import distance as dist # نحتاج لتثبيت scipy أو نستخدم دالة math

# --- إعدادات النظام الاحترافي ---
CONFIDENCE_THRESHOLD = 0.50   # دقة التعرف (توازن بين الشدة والمرونة)
EYE_ASPECT_RATIO_THRESHOLD = 0.25 # إذا نزل الرقم تحت هذا الحد، تعتبر العين مغلقة
CONSECUTIVE_FRAMES = 3        # عدد الفريمات للتأكد من الرمشة (لمنع الخطأ)
COOLDOWN_SECONDS = 60         # منع التكرار

# تجهيز مجلد الأدلة
EVIDENCE_DIR = "attendance_evidence"
if not os.path.exists(EVIDENCE_DIR):
    os.makedirs(EVIDENCE_DIR)

def get_eye_aspect_ratio(eye):
    """
    دالة رياضية لحساب نسبة فتحة العين.
    تعتمد على المسافة العمودية والافقية بين نقاط العين.
    """
    # حساب المسافات العمودية (بين الجفن العلوي والسفلي)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # حساب المسافة الأفقية (عرض العين)
    C = dist.euclidean(eye[0], eye[3])
    # المعادلة
    ear = (A + B) / (2.0 * C)
    return ear

def save_evidence(frame, name):
    """حفظ صورة الشخص لحظة التحضير كدليل"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{EVIDENCE_DIR}/{name}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"📸 Evidence saved: {filename}")

def main():
    print("--- 🛡️ Pro System: Liveness & Security (V3) ---")
    
    # تحميل البيانات
    users = db_manager.get_all_users()
    known_face_encodings = [user["encoding"] for user in users]
    known_face_names = [user["name"] for user in users]
    known_face_ids = [user["id"] for user in users]
    
    # متغيرات الحالة
    last_attendance = {}
    blink_counter = 0      # عداد الرمشات
    total_blinks = 0       # إجمالي الرمشات المكتملة
    is_eye_closed = False  # حالة العين

    video_capture = cv2.VideoCapture(0)
    print("🟢 النظام جاهز... (يجب أن ترمش لتسجيل الحضور!) 😉")

    while True:
        ret, frame = video_capture.read()
        if not ret: break

        # معالجة الصورة
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # 1. اكتشاف الوجه
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        # 2. استخراج معالم الوجه (العينين) لحساب الرمش
        # نحتاج للصورة الكبيرة للدقة في تحديد نقاط العين
        face_landmarks_list = face_recognition.face_landmarks(frame) 

        name = "Unknown"
        color = (0, 0, 255) # أحمر
        status_text = "Look at Camera"

        # إذا وجدنا وجهاً
        if len(face_encodings) > 0:
            # نأخذ أول وجه فقط للتبسيط
            face_encoding = face_encodings[0]
            
            # --- مرحلة التعرف (Identity Check) ---
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if face_distances[best_match_index] < CONFIDENCE_THRESHOLD:
                name = known_face_names[best_match_index]
                user_id = known_face_ids[best_match_index]
                
                # --- مرحلة كشف الحيوية (Liveness Check) ---
                # نتحقق هل وجدنا معالم للوجه في هذا الإطار؟
                if len(face_landmarks_list) > 0:
                    face_landmarks = face_landmarks_list[0]
                    left_eye = face_landmarks['left_eye']
                    right_eye = face_landmarks['right_eye']

                    # حساب نسبة فتحة العين
                    leftEAR = get_eye_aspect_ratio(left_eye)
                    rightEAR = get_eye_aspect_ratio(right_eye)
                    avgEAR = (leftEAR + rightEAR) / 2.0

                    # فحص الرمش
                    if avgEAR < EYE_ASPECT_RATIO_THRESHOLD:
                        blink_counter += 1
                        status_text = "Blinking..."
                    else:
                        # إذا كانت العين مغلقة لفترة كافية ثم فتحت -> هذه رمشة كاملة
                        if blink_counter >= CONSECUTIVE_FRAMES:
                            total_blinks += 1
                            is_eye_closed = True # تم اكتشاف رمشة حقيقية
                        blink_counter = 0
                        status_text = "Face Verified - Please Blink"

                # --- قرار التسجيل ---
                # الشرط: الوجه معروف + تم اكتشاف رمشة حقيقية واحدة على الأقل
                if is_eye_closed:
                    color = (0, 255, 0) # أخضر
                    status_text = f"Confirmed: {name}"
                    
                    current_time = time.time()
                    if user_id not in last_attendance or (current_time - last_attendance[user_id] > COOLDOWN_SECONDS):
                        # 1. تسجيل في القاعدة
                        db_manager.mark_attendance(user_id)
                        # 2. حفظ صورة الدليل
                        save_evidence(frame, name)
                        last_attendance[user_id] = current_time
                        
                        # إعادة تعيين الرمش للمرة القادمة
                        is_eye_closed = False 
                        total_blinks = 0
                        print(f"✅ Real Human Detected: {name}")

            else:
                status_text = "Unknown Person"

            # الرسم
            top, right, bottom, left = face_locations[0]
            top *= 4; right *= 4; bottom *= 4; left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, status_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, name, (left, bottom + 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        cv2.imshow('Security Attendance V3', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()