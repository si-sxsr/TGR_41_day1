import os
import csv
from PIL import Image

# กำหนด path ไปยังโฟลเดอร์ dataset ของคุณ
DATASET_DIR = "dataset"
OUTPUT_CSV = "image_manifest.csv"

# เตรียมลิสต์สำหรับเก็บข้อมูล
manifest_data = []

# วนลูปอ่านทุกโฟลเดอร์คลาส (good, black, husk, ฯลฯ)
for class_name in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, class_name)
    
    if os.path.isdir(class_path):
        # วนลูปอ่านไฟล์รูปภาพในแต่ละคลาส
        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_path, filename)
                
                # 1. source_image: ชื่อไฟล์
                source_image = filename
                
                # 2. class: ชื่อคลาส (มาจากชื่อโฟลเดอร์)
                # (ตัวแปร class_name มีค่าอยู่แล้ว)
                
                # 3. layout: ดึงจากชื่อไฟล์ (เช่น ดึง 'circle' จาก 'circle_black.jpg')
                layout = filename.split('_')[0]
                
                # 4 & 5. width, height: อ่านขนาดภาพด้วย PIL
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                except Exception as e:
                    print(f"อ่านไฟล์ภาพไม่ได้ {img_path}: {e}")
                    width, height = 0, 0
                
                # เก็บข้อมูลลงลิสต์
                manifest_data.append([source_image, class_name, layout, width, height])

# บันทึกข้อมูลลงไฟล์ CSV
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # เขียน Header
    writer.writerow(["source_image", "class", "layout", "width", "height"])
    # เขียนข้อมูล
    writer.writerows(manifest_data)

print(f"สร้างรายการไฟล์เสร็จสิ้น บันทึกไว้ที่ {OUTPUT_CSV}")