import cv2
import face_recognition
import numpy as np
from modules import db_manager
import time
import pyttsx3  # مكتبة الصوت

# --- إعدادات النظام (Configuration) ---
CONFIDENCE_THRESHOLD = 0.55  # (كلما قل الرقم، زاد التشدد في الدقة. 0.6 هو الافتراضي)
REQUIRED_FRAMES = 5          # عدد الفريمات المتتالية المطلوبة لتأكيد الهوية
COOLDOWN_SECONDS = 60        # الزمن بالثواني قبل السماح لنفس الشخص بالتسجيل مجدداً

# تجهيز الصوت
engine = pyttsx3.init()
engine.setProperty('rate', 100) # سرعة الكلام

def speak(text):
    """دالة نطق النصوص"""
    try:
        engine.say(text)
        engine.runAndWait()
    except:
        pass

def main():
    print("--- 🛡️ نظام الحضور الأمني المطور (V2) ---")
    
    # 1. تحميل البيانات
    users = db_manager.get_all_users()
    if not users:
        print("❌ قاعدة البيانات فارغة!")
        return

    known_face_encodings = [user["encoding"] for user in users]
    known_face_names = [user["name"] for user in users]
    known_face_ids = [user["id"] for user in users]
    
    # متغيرات للتحقق من الثبات (Stability Check)
    # الهيكل: { user_id: consecutive_frame_count }
    frame_counters = {}
    
    # متغير لتسجيل آخر وقت حضور (Cooldown)
    last_attendance = {}

    video_capture = cv2.VideoCapture(0)
    print("🟢 النظام جاهز... يرجى الثبات أمام الكاميرا.")

    while True:
        ret, frame = video_capture.read()
        if not ret: break

        # تصغير الصورة للمعالجة السريعة
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # قائمة الموظفين الموجودين في الفريم الحالي (لمعالجة العدادات)
        current_frame_users = []

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            user_id = None
            color = (0, 0, 255) # أحمر (مجهول) مبدئياً

            # حساب المسافة (الفرق) بين الوجه المكتشف والوجوه المحفوظة
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                best_score = face_distances[best_match_index]

                # التحقق الأمني: هل الشبه قوي بما يكفي؟
                if best_score < CONFIDENCE_THRESHOLD:
                    name = known_face_names[best_match_index]
                    user_id = known_face_ids[best_match_index]
                    current_frame_users.append(user_id)
                    
                    # --- منطق الاستقرار (Stability Logic) ---
                    # نبدأ العد: كم مرة رأينا هذا الوجه بشكل متتالي؟
                    frame_counters[user_id] = frame_counters.get(user_id, 0) + 1
                    
                    # نحدد اللون بناءً على حالة التحقق (أصفر: جاري التحقق، أخضر: تم التأكيد)
                    if frame_counters[user_id] < REQUIRED_FRAMES:
                        color = (0, 255, 255) # أصفر (اصبر قليلاً)
                        status_text = f"Verifying... {frame_counters[user_id]}/{REQUIRED_FRAMES}"
                    else:
                        color = (0, 255, 0) # أخضر (مؤكد)
                        status_text = "Confirmed"
                        
                        # --- مرحلة التسجيل ---
                        current_time = time.time()
                        if user_id not in last_attendance or (current_time - last_attendance[user_id] > COOLDOWN_SECONDS):
                            
                            # 1. تسجيل في القاعدة
                            db_manager.mark_attendance(user_id)
                            last_attendance[user_id] = current_time
                            
                            # 2. التفاعل الصوتي
                            print(f"✅ Welcome, {name}")
                            speak(f"Welcome {name}")
                        else:
                            status_text = "Already Marked"
                    
                    # رسم الحالة فوق المربع
                    cv2.putText(frame, status_text, (left*4, (top*4)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                else:
                    # وجه معروف لكن الشبه ضعيف (شخص يشبهه)
                    name = "Unknown"
                    frame_counters[best_match_index] = 0 # تصفير العداد

            # إعادة تكبير الإحداثيات والرسومات
            top *= 4; right *= 4; bottom *= 4; left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # تصفير عدادات الأشخاص الذين اختفوا من الكاميرا (لزيادة الدقة)
        # إذا اختفى وجهك، يجب أن تبدأ العد من جديد عند الظهور
        for uid in list(frame_counters.keys()):
            if uid not in current_frame_users:
                frame_counters[uid] = 0

        cv2.imshow('Pro Security Attendance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()