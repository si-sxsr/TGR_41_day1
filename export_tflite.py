"""
แปลง weights/best.keras -> weights/best.tflite สำหรับรันบน Raspberry Pi (ไม่ต้องลง TensorFlow เต็มตัว)

    python export_tflite.py            # float32 (ผลเท่า .keras)
    python export_tflite.py --quant    # dynamic-range INT8 (ไฟล์เล็กลง ~4 เท่า เร็วขึ้นบน CPU)
"""
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--src', default='weights/best.keras')
parser.add_argument('--dst', default='weights/best.tflite')
parser.add_argument('--quant', action='store_true', help='dynamic-range quantization (weights INT8)')
args = parser.parse_args()

model = tf.keras.models.load_model(args.src, compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
if args.quant:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open(args.dst, 'wb') as f:
    f.write(tflite_model)
print(f"บันทึก {args.dst} ({len(tflite_model) / 1e6:.2f} MB, quant={args.quant})")
