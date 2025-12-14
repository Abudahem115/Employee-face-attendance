import cv2
import face_recognition
import numpy as np
from modules import db_manager
import time
from scipy.spatial import distance as dist

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.users = db_manager.get_all_embeddings()
        self.known_face_encodings = [user["encoding"] for user in self.users]
        self.known_face_names = [user["name"] for user in self.users]
        self.known_face_ids = [user["id"] for user in self.users]
        
        self.last_attendance = {}
        self.blink_counter = 0
        self.consecutive_frames = 2
        self.eye_aspect_ratio_threshold = 0.23
        
        # متغيرات التحسين
        self.frame_counter = 0
        
        # --- ذاكرة الرسومات (الحل للمشكلة) ---
        # سنحفظ هنا آخر أماكن للوجوه لنرسمها في كل الفريمات
        self.last_locations = []
        self.last_names = []
        self.last_statuses = []
        self.last_colors = []

    def __del__(self):
        self.video.release()

    def get_eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def get_frame(self):
        success, frame = self.video.read()
        if not success: return None

        self.frame_counter += 1
        
        # نقوم بتحديث الذكاء الاصطناعي فقط كل 3 فريمات
        if self.frame_counter % 3 == 0:
            # تنظيف الذاكرة القديمة
            self.last_locations = []
            self.last_names = []
            self.last_statuses = []
            self.last_colors = []
            
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_small_frame)
            
            if len(face_locations) > 0:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, face_locations)
                
                # نفترض وجهاً واحداً للتبسيط
                face_encoding = face_encodings[0]
                face_loc = face_locations[0] # هذا الموقع للصورة الصغيرة
                
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                
                name = "Unknown"
                status_text = "Scanning..."
                color = (0, 255, 255) # أصفر

                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        user_id = self.known_face_ids[best_match_index]
                        
                        # منطق الرمش
                        is_blink = False
                        if len(face_landmarks_list) > 0:
                            landmarks = face_landmarks_list[0]
                            left_ear = self.get_eye_aspect_ratio(landmarks['left_eye'])
                            right_ear = self.get_eye_aspect_ratio(landmarks['right_eye'])
                            avg_ear = (left_ear + right_ear) / 2.0
                            
                            if avg_ear < self.eye_aspect_ratio_threshold:
                                self.blink_counter += 1
                            else:
                                if self.blink_counter >= self.consecutive_frames:
                                    is_blink = True
                                self.blink_counter = 0

                        if is_blink:
                            current_time = time.time()
                            if user_id not in self.last_attendance or (current_time - self.last_attendance[user_id] > 60):
                                db_manager.mark_attendance(user_id)
                                self.last_attendance[user_id] = current_time
                                status_text = f"WELCOME {name}"
                                color = (0, 255, 0)
                            else:
                                status_text = f"ALREADY MARKED"
                                color = (0, 255, 0)
                        else:
                            status_text = "PLEASE BLINK"
                            color = (0, 165, 255)

                # حفظ النتائج في الذاكرة لنستخدمها في الفريمات القادمة
                self.last_locations.append(face_loc)
                self.last_names.append(name)
                self.last_statuses.append(status_text)
                self.last_colors.append(color)

        # --- قسم الرسم (يعمل في كل فريم باستخدام الذاكرة) ---
        for (top, right, bottom, left), name, status, color in zip(self.last_locations, self.last_names, self.last_statuses, self.last_colors):
            # تكبير الإحداثيات ×4 لأننا حسبناها على الصورة الصغيرة
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, status, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, name, (left, bottom + 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()