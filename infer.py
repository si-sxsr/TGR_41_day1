"""
Inference สำหรับ Day 1 (Coffee Bean Classification) ตาม INFER_SPEC

    python infer.py --input test/ --weights weights/ --out outputs/

ขั้นตอน: แยกเมล็ดด้วย OpenCV (ลบพื้นหลัง + watershed + ตัดหน้าต่าง สำหรับเมล็ดที่ติดกันในกอง)
-> crop รายเมล็ด -> MobileNetV3-Small -> เฉลี่ยความน่าจะเป็นทุกเมล็ด -> 1 คลาสต่อภาพ
"""
import os
import sys
import csv
import json
import shutil
import argparse

import cv2
import numpy as np
from PIL import Image, ImageOps

CLASSES = ['black', 'broken', 'good', 'husk', 'immature', 'parchment', 'shell']  # ลำดับ A-Z ตามตอนเทรน
OUTPUT_CLASSES = ['good', 'shell', 'black', 'husk', 'broken', 'parchment', 'immature']
IMG_EXTS = ('.jpg', '.jpeg', '.png')
IMG_SIZE = (224, 224)

# พารามิเตอร์แยกเมล็ด (คิดบนภาพย่อ SCALE)
SCALE = 0.5
MIN_AREA = 2000        # เล็กกว่านี้คือ noise
SINGLE_MAX_AREA = 15000  # ใหญ่กว่านี้ถือว่าเป็นกลุ่มเมล็ดติดกัน -> watershed
PAD = 10               # padding รอบเมล็ด (พิกเซลภาพจริง)
WB_TARGET = 200.0      # สีกระดาษหลัง white balance
FG_THRESH = 30        # เมล็ดต้องมืดกว่า/สีอิ่มกว่ากระดาษเกินค่านี้


class InferError(Exception):
    pass


# ----------------------------------------------------------------------------
# อ่านภาพ + แยกเมล็ด (ใช้ร่วมกับ prepare_crops.py เพื่อให้ train/infer ตรงกัน)
# ----------------------------------------------------------------------------
def read_image_bgr(path):
    """อ่านภาพพร้อมแก้ EXIF orientation คืนค่า BGR uint8"""
    try:
        pil_img = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    except Exception as e:
        raise InferError(f"อ่านภาพไม่ได้: {path} ({e})")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _split_clump(blob_mask, radius):
    """แยก blob ที่มีหลายเมล็ดติดกัน: จุดยอดของ distance transform = จุดกลางเมล็ด -> watershed"""
    dist = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
    k = max(3, int(1.2 * radius) | 1)
    peaks = (dist >= cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))) & (dist > 0.4 * radius)
    n, markers = cv2.connectedComponents(peaks.astype(np.uint8))
    if n <= 2:  # แยกไม่ออก
        return [blob_mask]
    markers = markers + 1
    markers[blob_mask == 0] = 1          # พื้นหลัง
    markers[(blob_mask > 0) & ~peaks] = 0  # ให้ watershed ตัดสิน
    ws = cv2.watershed(cv2.cvtColor(blob_mask * 255, cv2.COLOR_GRAY2BGR), markers)
    return [((ws == lbl) & (blob_mask > 0)).astype(np.uint8) for lbl in range(2, n + 1)]


def foreground_mask(small):
    """
    หาเมล็ดด้วยการลบพื้นหลัง: ประมาณความสว่างกระดาษด้วย closing ขนาดใหญ่ (ทนต่อแสงไม่สม่ำเสมอ)
    เมล็ด = มืดกว่ากระดาษ หรือ สีอิ่มกว่ากระดาษ (เมล็ดสีอ่อนอย่าง parchment)
    """
    h, w = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    q = 4
    g_low = cv2.resize(gray, (w // q, h // q), interpolation=cv2.INTER_AREA)
    bg = cv2.morphologyEx(g_low, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)))
    bg = cv2.resize(cv2.GaussianBlur(bg, (31, 31), 0), (w, h))
    fg = ((bg - gray) > FG_THRESH) | ((sat - np.median(sat)) > FG_THRESH)
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def extract_bean_crops(img):
    """
    คืน list ของ crop รายเมล็ด (BGR, ขนาดตามจริง) โดยพื้นที่นอกเมล็ดถูกแทนด้วยสีพื้นหลัง
    เพื่อไม่ให้เมล็ดข้างเคียงในกองปนเข้ามาใน crop
    """
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * SCALE), int(h * SCALE)), interpolation=cv2.INTER_AREA)
    mask = foreground_mask(small)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = areas[areas > MIN_AREA]
    # เกณฑ์ "เมล็ดเดี่ยว" ปรับตามขนาดเมล็ดในภาพ (ภาพถ่ายใกล้ เมล็ดใหญ่กว่าปกติ)
    single_max = max(SINGLE_MAX_AREA, 2.0 * float(np.median(big))) if len(big) >= 4 else SINGLE_MAX_AREA
    singles = big[big < single_max]
    # รัศมีเมล็ดโดยประมาณ (ใช้เมล็ดเดี่ยวในภาพ ถ้าไม่มีใช้ค่ามาตรฐาน)
    radius = float(np.sqrt(np.median(singles) / np.pi)) if len(singles) >= 3 else 50.0

    regions = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= MIN_AREA:
            continue
        blob = (labels == i).astype(np.uint8)
        if area < single_max:
            regions.append(blob)
        else:
            regions.extend(p for p in _split_clump(blob, radius) if p.sum() > MIN_AREA)

    # สีพื้นหลัง = ค่ากลางของพิกเซลที่ไม่ใช่เมล็ด
    bg_raw = np.median(small[mask == 0], axis=0) if (mask == 0).any() else np.array([200.0, 200.0, 200.0])
    wb_gain = (WB_TARGET / np.maximum(bg_raw, 1.0)).astype(np.float32)
    bg_color = np.array([WB_TARGET] * 3, np.uint8)
    dil_kernel = np.ones((7, 7), np.uint8)

    single_area = np.pi * radius ** 2
    crops = []
    for region in regions:
        x, y, cw, ch = cv2.boundingRect(region)
        if region.sum() > 2.2 * single_area:
            # ยังเป็นกลุ่มหลายเมล็ด: ตัดเป็นหน้าต่างขนาดราวหนึ่งเมล็ด
            boxes = _tile_boxes(region, x, y, cw, ch, radius)
        else:
            ar = cw / ch if ch else 0
            if not (0.3 <= ar <= 3.0):
                continue
            boxes = [(x, y, cw, ch)]
        m_dil = cv2.dilate(region, dil_kernel)
        for bx, by, bw, bh in boxes:
            x1, y1 = max(0, int(bx / SCALE) - PAD), max(0, int(by / SCALE) - PAD)
            x2, y2 = min(w, int((bx + bw) / SCALE) + PAD), min(h, int((by + bh) / SCALE) + PAD)
            crop = img[y1:y2, x1:x2].astype(np.float32)
            if crop.size == 0:
                continue
            # white balance: ปรับให้สีกระดาษเป็นเทากลาง (ลดผลจากกล้อง/แสงต่างกัน)
            crop = np.clip(crop * wb_gain, 0, 255).astype(np.uint8)
            # พื้นที่นอก region (ขยายขอบเล็กน้อย) แทนด้วยสีพื้นหลัง
            m_small = m_dil[int(y1 * SCALE):int(np.ceil(y2 * SCALE)), int(x1 * SCALE):int(np.ceil(x2 * SCALE))]
            m_full = cv2.resize(m_small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            crop[m_full == 0] = bg_color
            crops.append(crop)
    return crops


def _tile_boxes(region, x, y, cw, ch, radius):
    """หน้าต่างสี่เหลี่ยมขนาด ~1 เมล็ด เลื่อนครึ่งหน้าต่าง เก็บเฉพาะที่มีเมล็ดเกิน 35%"""
    s = int(2.6 * radius)
    step = max(1, s // 2)
    boxes = []
    for yy in range(y, max(y + 1, y + ch - s // 2), step):
        for xx in range(x, max(x + 1, x + cw - s // 2), step):
            win = region[yy:yy + s, xx:xx + s]
            if win.size and win.mean() > 0.35:
                boxes.append((xx, yy, win.shape[1], win.shape[0]))
    return boxes or [(x, y, cw, ch)]


def preprocess_crop(crop_bgr):
    """crop BGR -> RGB float32 224x224 ช่วง 0-255 (MobileNetV3 ของ Keras มี Rescaling ในตัว)"""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR).astype(np.float32)


# ----------------------------------------------------------------------------
# Interface ตามแม่แบบ: load_model() และ predict_image()
# ----------------------------------------------------------------------------
def _tflite_interpreter_cls():
    """หา TFLite runtime ที่ติดตั้งไว้: ai-edge-litert (แนะนำ) -> tflite-runtime -> tensorflow"""
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    try:
        import tensorflow as tf
        return tf.lite.Interpreter
    except ImportError:
        raise InferError("ไม่พบ TFLite runtime: pip install ai-edge-litert")


class TFLiteModel:
    def __init__(self, path):
        self.interp = _tflite_interpreter_cls()(model_path=path, num_threads=os.cpu_count() or 4)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]['index']
        self.out = self.interp.get_output_details()[0]['index']

    def predict(self, batch):
        outs = []
        for x in batch:  # โมเดลรับ batch 1
            self.interp.set_tensor(self.inp, x[None].astype(np.float32))
            self.interp.invoke()
            outs.append(self.interp.get_tensor(self.out)[0].copy())
        return np.array(outs)


class KerasModel:
    def __init__(self, path):
        import tensorflow as tf  # import ช้า จึงโหลดเมื่อจำเป็น
        self.model = tf.keras.models.load_model(path, compile=False)

    def predict(self, batch):
        return self.model.predict(batch, verbose=0)


def load_model(weights_dir):
    """ใช้ weights/best.tflite ถ้ามี (เบา เหมาะกับ Pi) ไม่งั้นใช้ weights/best.keras"""
    classes = CLASSES
    map_path = os.path.join(weights_dir, 'class_mapping.json')
    if os.path.isfile(map_path):
        with open(map_path, encoding='utf-8') as f:
            mapping = json.load(f)
        classes = [mapping[str(i)] for i in range(len(mapping))]

    tflite_path = os.path.join(weights_dir, 'best.tflite')
    keras_path = os.path.join(weights_dir, 'best.keras')
    if os.path.isfile(tflite_path):
        return TFLiteModel(tflite_path), classes
    if os.path.isfile(keras_path):
        return KerasModel(keras_path), classes
    raise InferError(f"ไม่พบ weights (best.tflite หรือ best.keras) ใน {weights_dir}")


def predict_image(model_bundle, img_path):
    model, classes = model_bundle
    img = read_image_bgr(img_path)
    crops = extract_bean_crops(img)
    if not crops:
        # หาเมล็ดไม่เจอ: ใช้ทั้งภาพแทน (ยังได้ผล 1 คลาสต่อภาพ)
        crops = [img]
    batch = np.stack([preprocess_crop(c) for c in crops])
    probs = model.predict(batch)
    return classes[int(np.argmax(probs.mean(axis=0)))]


# ----------------------------------------------------------------------------
def list_images(input_path):
    if os.path.isfile(input_path):
        if not input_path.lower().endswith(IMG_EXTS):
            raise InferError(f"ไฟล์ input ไม่ใช่ภาพที่รองรับ: {input_path}")
        return [input_path]
    if os.path.isdir(input_path):
        files = sorted(f for f in os.listdir(input_path) if f.lower().endswith(IMG_EXTS))
        if not files:
            raise InferError(f"ไม่พบภาพ (.jpg/.jpeg/.png) ใน {input_path}")
        return [os.path.join(input_path, f) for f in files]
    raise InferError(f"ไม่พบ input: {input_path}")


def main():
    parser = argparse.ArgumentParser(description="Coffee bean classification - Day 1 inference")
    parser.add_argument('--input', required=True, help='โฟลเดอร์ภาพทดสอบ หรือไฟล์ภาพ')
    parser.add_argument('--weights', required=True, help='โฟลเดอร์ weights')
    parser.add_argument('--out', required=True, help='โฟลเดอร์บันทึกผล')
    args = parser.parse_args()

    images = list_images(args.input)
    if not os.path.isdir(args.weights):
        raise InferError(f"ไม่พบโฟลเดอร์ weights: {args.weights}")
    bundle = load_model(args.weights)

    os.makedirs(args.out, exist_ok=True)
    for c in OUTPUT_CLASSES:
        os.makedirs(os.path.join(args.out, c), exist_ok=True)

    rows = []
    for path in images:
        name = os.path.basename(path)
        pred = predict_image(bundle, path)
        if pred not in OUTPUT_CLASSES:
            raise InferError(f"คลาสที่ทำนายไม่อยู่ในรายการ: {pred}")
        shutil.copy2(path, os.path.join(args.out, pred, name))  # คัดลอกไฟล์ต้นฉบับ ไม่แก้ไข
        rows.append((name, pred))
        print(f"{name},{pred}")

    if len({r[0] for r in rows}) != len(rows):
        raise InferError("พบชื่อไฟล์ซ้ำใน input")

    # สเปก INFER_SPEC: predictions.csv (filename,class)
    # สไลด์หน้า 22: prediction.csv (image_name,prediction) — เขียนทั้งสองแบบ กันกรรมการใช้คนละแบบ
    for fname, header in (('predictions.csv', ['filename', 'class']),
                          ('prediction.csv', ['image_name', 'prediction'])):
        with open(os.path.join(args.out, fname), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
    print(f"เสร็จ {len(rows)} ภาพ -> {os.path.join(args.out, 'predictions.csv')}")


if __name__ == '__main__':
    try:
        main()
    except InferError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
