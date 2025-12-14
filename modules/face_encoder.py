import face_recognition
import pickle
import os

def get_face_encoding(image_path):
    
    if not os.path.exists(image_path):
        print(f"❌ Error: The image is not in the path:{image_path}")
        return None

    try:
        print(f"⏳ Image analysis in progress: {image_path}...")
        
        image = face_recognition.load_image_file(image_path)
        
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            face_encoding = encodings[0]
            
            encoding_bytes = pickle.dumps(face_encoding)
            
            print("✅ The facial fingerprint was successfully extracted!")
            return encoding_bytes
        else:
            print("⚠️ No face was found in the picture (try a clearer picture).")
            return None

    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":

    test_image_path = "../me.jpg" 
    
    print("--- AI Unit Test---")
    if os.path.exists(test_image_path):
        result = get_face_encoding(test_image_path)
        if result:
            print(f"Success! Extracted data size: {len(result)} bytes")
    else:
        print(f"To test this: Place an image named me.jpg outside the modules folder and run this file.")