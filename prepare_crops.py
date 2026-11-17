"""
ตัดภาพเมล็ดรายเม็ดจาก dataset/<class>/<layout>_<class>.jpg ไปไว้ที่ crops_dataset/

    python prepare_crops.py grid     # fold: grid เป็น val ที่เหลือเป็น train
    python prepare_crops.py circle
    python prepare_crops.py square
    python prepare_crops.py all      # ใช้ทุกภาพเป็น train (โมเดลตัวจริงสำหรับส่ง)

ใช้ฟังก์ชันแยกเมล็ดตัวเดียวกับ infer.py เพื่อให้ crop ตอนเทรนกับตอน inference เหมือนกัน
"""
import os
import sys
import shutil

import cv2
import numpy as np

from infer import read_image_bgr, extract_bean_crops
from synth_piles import make_piles

DATASET_DIR = "dataset"
OUTPUT_CROPS_DIR = "crops_dataset"
CLASSES = ['good', 'shell', 'black', 'husk', 'broken', 'parchment', 'immature']
LAYOUTS = ['circle', 'square', 'grid']
PILES_PER_IMAGE = 3          # ภาพกองจำลองต่อภาพต้นฉบับ (เพิ่ม crop แบบเมล็ดชิดกัน)
VAL_PILES_DIR = "val_piles"  # ภาพกองจำลองจาก layout ที่เป็น val ใช้วัดผลระดับภาพด้วย infer.py


def main():
    val_layout = sys.argv[1] if len(sys.argv) > 1 else 'grid'
    if val_layout == 'all':
        train_layouts, val_layouts = LAYOUTS, []
    elif val_layout in LAYOUTS:
        train_layouts, val_layouts = [l for l in LAYOUTS if l != val_layout], [val_layout]
    else:
        sys.exit(f"ใช้ได้เฉพาะ {LAYOUTS + ['all']}")
    train_layouts = train_layouts + ['real']  # ภาพถ่ายจริงเพิ่มเติม (real_<class>_*.jpg) ใช้เทรนเสมอ
    print(f"Train: {train_layouts}  Val: {val_layouts}")

    shutil.rmtree(OUTPUT_CROPS_DIR, ignore_errors=True)  # ลบ crop เก่าก่อน กันไฟล์ค้าง
    shutil.rmtree(VAL_PILES_DIR, ignore_errors=True)
    rng = np.random.default_rng(0)

    for class_name in CLASSES:
        class_path = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
        for filename in sorted(os.listdir(class_path)):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            layout = filename.split('_')[0]
            if layout in train_layouts:
                split = "train"
            elif layout in val_layouts:
                split = "val"
            else:
                continue
            out_dir = os.path.join(OUTPUT_CROPS_DIR, split, class_name)
            os.makedirs(out_dir, exist_ok=True)

            img = read_image_bgr(os.path.join(class_path, filename))
            stem = os.path.splitext(filename)[0]
            crops = extract_bean_crops(img)
            for i, crop in enumerate(crops):
                cv2.imwrite(os.path.join(out_dir, f"{stem}_bean_{i}.jpg"), crop)

            n_pile_crops = 0
            for k, pile in enumerate(make_piles(img, rng, PILES_PER_IMAGE)):
                if split == "val":
                    os.makedirs(VAL_PILES_DIR, exist_ok=True)
                    cv2.imwrite(os.path.join(VAL_PILES_DIR, f"pile{k}_{stem}.jpg"), pile, [cv2.IMWRITE_JPEG_QUALITY, 92])
                for i, crop in enumerate(extract_bean_crops(pile)):
                    cv2.imwrite(os.path.join(out_dir, f"{stem}_pile{k}_{i}.jpg"), crop)
                    n_pile_crops += 1
            print(f"[{split}] {filename} -> {len(crops)} beans + {n_pile_crops} pile crops")


if __name__ == "__main__":
    main()
