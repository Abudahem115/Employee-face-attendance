import cv2
import face_recognition
import numpy as np
from modules import db_manager
import time

def main():
    print("--- 📷 Smart Attendance System Operation ---")
    
    # 1. تحميل البيانات
    users = db_manager.get_all_users()
    if not users:
        print("❌ No employees!")
        return

    known_face_encodings = [user["encoding"] for user in users]
    known_face_names = [user["name"] for user in users]
    known_face_ids = [user["id"] for user in users] # نحتاج الـ ID للتسجيل
    
    # قاموس لحفظ وقت آخر تسجيل لكل شخص لمنع التكرار
    # الشكل: { user_id: time_of_last_scan }
    last_attendance = {}

    video_capture = cv2.VideoCapture(0)
    print("🟢 The system is working... (Press 'q' to exit)")

    while True:
        ret, frame = video_capture.read()
        if not ret: break

        # تصغير ومعالجة الصورة
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            user_id = None

            # المطابقة
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    user_id = known_face_ids[best_match_index]

                    # --- منطق تسجيل الحضور ---
                    current_time = time.time()
                    
                    # إذا لم يسجل من قبل، أو مر وقت كافٍ (مثلاً 60 ثانية) على آخر تسجيل
                    if user_id not in last_attendance or (current_time - last_attendance[user_id] > 60):
                        db_manager.mark_attendance(user_id)
                        last_attendance[user_id] = current_time
                        print(f"✅ Attendance has been recorded: {name}")
                    # ---------------------------

            # الرسم على الشاشة
            top *= 4; right *= 4; bottom *= 4; left *= 4
            
            # تغيير اللون: أخضر إذا تم التسجيل حديثاً، أزرق إذا كان مجرد مشاهدة
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow('Smart Attendance System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()