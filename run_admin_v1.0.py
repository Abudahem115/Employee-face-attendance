import os
from modules import db_manager
from modules import face_encoder

def main():
    print("\n--- 👤 Employee Registration System (Admin Tool) ---")
    
    # 1. تهيئة الجداول
    db_manager.init_tables()
    
    while True:
        # 2. طلب البيانات
        name = input("\nEnter the employee's name: ").strip()
        image_path = input("Enter the image filename (e.g., me.jpg): ").strip()
        
        # 3. التحقق والمعالجة
        if os.path.exists(image_path):
            print("⏳ Image analysis is underway...")
            face_data = face_encoder.get_face_encoding(image_path)
            
            if face_data:
                # 4. الحفظ في قاعدة البيانات
                if db_manager.add_user(name, face_data):
                    print(f"🎉 Employee '{name}' has been successfully saved!")
            else:
                print("❌ No clear face was found in the picture.")
        else:
            print("❌ The image file does not exist!")

        # خيار الاستمرار
        if input("\nDo you want to add another employee? (y/n)").lower() != 'y':
            break
            
    print("The system has been logged out.")

if __name__ == "__main__":
    main()