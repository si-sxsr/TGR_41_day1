# TGR-41 · Day 1 · Coffee Bean Classification

โมเดลแยกคุณภาพเมล็ดกาแฟ 7 คลาส: `good`, `shell`, `black`, `husk`, `broken`, `parchment`, `immature`
หนึ่งภาพ (เมล็ดคลาสเดียว วางกอง) → หนึ่งคลาส

## วิธีการ (Bean-crop classification)

1. **แยกเมล็ด (OpenCV):** ประมาณความสว่างของพื้นกระดาษ แล้วถือว่าพิกเซลที่มืดกว่าหรือสีอิ่มกว่ากระดาษเป็นเมล็ด
   - เมล็ดที่ติดกันเป็นกอง: แยกด้วย distance transform + watershed
   - ถ้ายังแยกไม่ออก: ตัดเป็นหน้าต่างขนาดราวหนึ่งเมล็ด
2. **Crop รายเมล็ด:** ปรับ white balance ให้สีกระดาษเป็นเทากลาง (ลดผลจากกล้อง/แสงที่ต่างกัน) แล้วแทนพื้นที่นอกเมล็ดด้วยสีพื้นหลัง กันเมล็ดข้างเคียงปนเข้ามา เกณฑ์ขนาดเมล็ดปรับตามภาพ จึงรองรับภาพที่ถ่ายใกล้/ไกลต่างกัน
3. **จำแนก:** MobileNetV3-Small (ImageNet pretrained, fine-tune) ขนาด input 224×224
4. **รวมผล:** เฉลี่ยความน่าจะเป็นของทุกเมล็ดในภาพ แล้วเลือกคลาสที่สูงสุด

ภาพ train จัดเรียงเมล็ดไม่ชนกัน แต่ภาพ test วางเป็นกอง จึงสร้าง **ภาพกองจำลอง** ตอนเทรน (`synth_piles.py`) โดย copy-paste เมล็ดให้ชิดและซ้อนกันบางส่วน แล้วตัด crop ด้วยวิธีเดียวกับตอน inference

## ติดตั้ง (สำหรับ inference)

ใช้ Python 3.9–3.12 บน Raspberry Pi OS **64-bit**

```sh
pip install -r requirements.txt
```

`infer.py` ใช้ `weights/best.tflite` ผ่าน `ai-edge-litert` จึงไม่ต้องติดตั้ง TensorFlow
ถ้าไม่มี `best.tflite` โปรแกรมจะใช้ `weights/best.keras` แทน ซึ่งต้องติดตั้ง `tensorflow>=2.16`

## รัน inference

```sh
python infer.py --input <test_folder/> --weights weights/ --out outputs/
```

- `--input` ใส่ได้ทั้งโฟลเดอร์ภาพหรือไฟล์ภาพเดียว รองรับ `.jpg`, `.jpeg`, `.png` (ไม่สนตัวพิมพ์เล็ก/ใหญ่)
- ไม่ต้องรัน preprocessing แยก เพราะ `infer.py` แก้ EXIF orientation และแยกเมล็ดเองทั้งหมด
- ไม่มี GUI ไม่มี test-time augmentation
- ถ้า input/weights ไม่พบ ไม่มีภาพ หรืออ่านภาพไม่ได้ จะแสดง error และจบด้วย exit code 1

### ผลลัพธ์

```
outputs/
├── predictions.csv   # filename,class         (ตาม INFER_SPEC)
├── prediction.csv    # image_name,prediction  
├── good/ shell/ black/ husk/ broken/ parchment/ immature/   # สำเนาภาพต้นฉบับตามคลาสที่ทำนาย
```

## ไฟล์ใน weights/

| ไฟล์ | ใช้ทำอะไร |
|---|---|
| `best.tflite` | โมเดลหลักที่ `infer.py` ใช้ (float32) |
| `best.keras` | โมเดลต้นฉบับ Keras ใช้แทนได้ถ้าไม่มีไฟล์ `.tflite` |
| `class_mapping.json` | ลำดับ index → ชื่อคลาส |

## เทรนใหม่ (ไม่จำเป็นสำหรับการตรวจ)

ต้องติดตั้งเพิ่ม: `tensorflow>=2.16`, `scikit-learn`

ไฟล์ที่ส่งมีเฉพาะ `train.py` และ `infer.py` ตามโครงสร้างที่กำหนด สคริปต์เตรียมข้อมูล (`prepare_crops.py`, `synth_piles.py`, `export_tflite.py`, `prepare_data.py`) อยู่ใน repo ของทีม ไม่ได้อยู่ในไฟล์ส่ง ขั้นตอนเทรนเต็มคือ:

```sh
python prepare_data.py          # (ไม่บังคับ) สร้าง image_manifest.csv
python prepare_crops.py all     # ตัด crop รายเมล็ด + crop จากภาพกองจำลอง -> crops_dataset/
python train.py                 # เทรน -> weights/best.keras
python export_tflite.py         # แปลง -> weights/best.tflite
```

ข้อมูลต้องวางไว้ที่ `dataset/<class>/<layout>_<class>.jpg` เช่น `dataset/good/circle_good.jpg`
ภาพถ่ายเพิ่มเติมตั้งชื่อเป็น `dataset/<class>/real_<class>_<ชื่อใดก็ได้>.jpg` จะถูกใช้เทรนเสมอ

ถ้าต้องการวัดผลแบบ fold ให้ใช้ `python prepare_crops.py grid` (หรือ `circle`, `square`) แทน `all`
- layout ที่เลือกจะถูกใช้เป็น val ส่วนที่เหลือเป็น train
- จะได้ภาพกองจำลองจาก layout นั้นที่ `val_piles/` ไว้วัดผลระดับภาพด้วย `infer.py`

## ผลการทดลอง

ทดสอบด้วยการแบ่ง fold ตาม layout (val คือ layout ที่โมเดลไม่เคยเห็นตอนเทรน)

| การทดลอง | ภาพแยกเมล็ด (val) | ภาพกองจำลอง (val) |
|---|---|---|
| เทรนเฉพาะ crop เมล็ดเดี่ยว (fold grid) | 6/7 | 9/21 |
| เทรนรวม crop จากภาพกองจำลอง (fold grid) | 6/7 | **19/21** |
| ผลเดียวกันหลังแปลงเป็น TFLite float32 | – | 19/21 (ผลตรงกับ `.keras` ทุกภาพ) |

คลาสที่ยังสับสนมากที่สุดคือ `immature` ซึ่งมักถูกทายเป็น `parchment` หรือ `good` เพราะสีใกล้กัน

### โมเดลที่ส่ง

เทรนด้วยข้อมูลทั้ง 3 layout ของงานแข่ง (`prepare_crops.py all`) จำนวน 12 epoch รวมกับภาพที่ทีมถ่ายเพิ่มอีก 11 ภาพ (black, broken, good, shell)

ทดสอบกับภาพที่ทีมถ่ายเอง ซึ่งใช้กล้องและแสงต่างจากชุดของงานแข่ง:

| โมเดล | ภาพถ่ายเองที่ไม่ได้ใช้เทรน (4 ภาพ) | ภาพชุดงานแข่ง (21 ภาพ) | ภาพกองจำลอง (21 ภาพ) |
|---|---|---|---|
| ไม่ใส่ภาพถ่ายเอง | 1/4 | 21/21 | 21/21 |
| **ใส่ภาพถ่ายเอง (ตัวที่ส่ง)** | **4/4** | **21/21** | **21/21** |

ภาพชุดงานแข่งและภาพกองจำลองถูกใช้เทรนด้วย ตัวเลขสองคอลัมน์หลังจึงใช้ยืนยันว่าผลไม่ตกลง ไม่ได้วัดความแม่นจริง
