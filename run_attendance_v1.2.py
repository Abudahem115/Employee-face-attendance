import cv2
import face_recognition
import numpy as np
from modules import db_manager
import time
import os
from datetime import datetime
from scipy.spatial import distance as dist

CONFIDENCE_THRESHOLD = 0.50   
EYE_ASPECT_RATIO_THRESHOLD = 0.25 
CONSECUTIVE_FRAMES = 3       
COOLDOWN_SECONDS = 60       

EVIDENCE_DIR = "attendance_evidence"
if not os.path.exists(EVIDENCE_DIR):
    os.makedirs(EVIDENCE_DIR)

def get_eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def save_evidence(frame, name):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{EVIDENCE_DIR}/{name}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"📸 Evidence saved: {filename}")

def main():
    print("--- 🛡️ Pro System: Liveness & Security (V3) ---")
    
    users = db_manager.get_all_users()
    known_face_encodings = [user["encoding"] for user in users]
    known_face_names = [user["name"] for user in users]
    known_face_ids = [user["id"] for user in users]
    
    last_attendance = {}
    blink_counter = 0      
    total_blinks = 0      
    is_eye_closed = False  

    video_capture = cv2.VideoCapture(0)
    print("🟢 The express system is ready... (Blink to register attendance!) 😉")

    while True:
        ret, frame = video_capture.read()
        if not ret: break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_landmarks_list = face_recognition.face_landmarks(frame) 

        name = "Unknown"
        color = (0, 0, 255) 
        status_text = "Look at Camera"

        if len(face_encodings) > 0:
            face_encoding = face_encodings[0]
            
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if face_distances[best_match_index] < CONFIDENCE_THRESHOLD:
                name = known_face_names[best_match_index]
                user_id = known_face_ids[best_match_index]
                
                if len(face_landmarks_list) > 0:
                    face_landmarks = face_landmarks_list[0]
                    left_eye = face_landmarks['left_eye']
                    right_eye = face_landmarks['right_eye']

                    leftEAR = get_eye_aspect_ratio(left_eye)
                    rightEAR = get_eye_aspect_ratio(right_eye)
                    avgEAR = (leftEAR + rightEAR) / 2.0

                    if avgEAR < EYE_ASPECT_RATIO_THRESHOLD:
                        blink_counter += 1
                        status_text = "Blinking..."
                    else:
                        if blink_counter >= CONSECUTIVE_FRAMES:
                            total_blinks += 1
                            is_eye_closed = True 
                        blink_counter = 0
                        status_text = "Face Verified - Please Blink"

                if is_eye_closed:
                    color = (0, 255, 0) 
                    status_text = f"Confirmed: {name}"
                    
                    current_time = time.time()
                    if user_id not in last_attendance or (current_time - last_attendance[user_id] > COOLDOWN_SECONDS):
                        db_manager.mark_attendance(user_id)
                        save_evidence(frame, name)
                        last_attendance[user_id] = current_time
                        
                        is_eye_closed = False 
                        total_blinks = 0
                        print(f"✅ Real Human Detected: {name}")

            else:
                status_text = "Unknown Person"

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