import cv2
import face_recognition
import os
import numpy as np
from modules import db_manager
import time

def main():
    print("\n--- 👤 Smart Employee Registration (Live Capture) ---")
    
    # تهيئة قاعدة البيانات الجديدة
    db_manager.init_db()
    
    name = input("Enter Employee Name: ").strip()
    if not name:
        print("❌ Name cannot be empty.")
        return

    print(f"\n🎥 Opening camera for {name}...")
    print("Please rotate your head slightly (Left, Right, Center) to capture angles.")
    
    video_capture = cv2.VideoCapture(0)
    
    captured_encodings = []
    REQUIRED_SAMPLES = 15  # عدد الصور المطلوب التقاطها (كلما زاد، زادت الدقة)
    
    while len(captured_encodings) < REQUIRED_SAMPLES:
        ret, frame = video_capture.read()
        if not ret: break
        
        # نسخة للعرض
        display_frame = frame.copy()
        
        # تحويل الألوان
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # كشف الوجه
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if len(face_locations) == 1:
            # تم العثور على وجه واحد بالضبط (ممتاز)
            top, right, bottom, left = face_locations[0]
            
            # رسم مربع أخضر
            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # استخراج البصمة
            try:
                face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                captured_encodings.append(face_encoding)
                
                # وميض أبيض بسيط (Visual Feedback)
                cv2.rectangle(display_frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 255), 10)
                
                print(f"📸 Captured {len(captured_encodings)}/{REQUIRED_SAMPLES}")
                time.sleep(0.1) # انتظار بسيط بين الصور
            except:
                pass
                
        elif len(face_locations) > 1:
            cv2.putText(display_frame, "One person only!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "Looking for face...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # شريط التقدم
        progress = f"Progress: {len(captured_encodings)}/{REQUIRED_SAMPLES}"
        cv2.putText(display_frame, progress, (50, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow('Registration Mode', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("❌ Cancelled by user.")
            return

    video_capture.release()
    cv2.destroyAllWindows()
    
    # الحفظ في قاعدة البيانات
    if len(captured_encodings) == REQUIRED_SAMPLES:
        print("\n💾 Saving data to database...")
        db_manager.add_user_with_encodings(name, captured_encodings)
        print("🎉 Registration Complete! System is now trained on your face.")
    else:
        print("❌ Registration failed/incomplete.")

if __name__ == "__main__":
    main()