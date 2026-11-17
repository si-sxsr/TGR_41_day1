"""
เทรน MobileNetV3-Small บน crop รายเมล็ด

    python prepare_crops.py grid && python train.py   # ทดลอง fold (มี val, เก็บ epoch ที่ดีที่สุด)
    python prepare_crops.py all  && python train.py   # โมเดลตัวจริง (ไม่มี val, เทรนจำนวน epoch คงที่)
"""
import os
import json
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 40          # โหมดมี val (EarlyStopping จะหยุดเอง)
FINAL_EPOCHS = 12    # โหมดไม่มี val: ค่าเฉลี่ย epoch ที่ดีที่สุดจาก 3 fold
NUM_CLASSES = 7
TRAIN_DIR, VAL_DIR = 'crops_dataset/train', 'crops_dataset/val'
# มี val ก็ต่อเมื่อมีไฟล์ภาพจริง (โฟลเดอร์ว่างที่ค้างจากรอบก่อนไม่นับ)
has_val = any(f.lower().endswith(('.jpg', '.jpeg', '.png'))
              for _, _, files in os.walk(VAL_DIR) for f in files)

# ไม่ใช้ rescale: MobileNetV3 ของ Keras มี Rescaling ในตัว ต้องการพิกเซล 0-255
# interpolation='bilinear' ให้ตรงกับ cv2.INTER_LINEAR ใน infer.py
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=360,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    zoom_range=0.2,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=True, interpolation='bilinear')

val_generator = None
if has_val:
    val_generator = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False, interpolation='bilinear')

# Class weights กันโมเดลลำเอียงไปคลาสที่มีเมล็ดเยอะ
y_train = train_generator.classes
class_weights = dict(enumerate(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))
print("Class weights:", class_weights)

base_model = MobileNetV3Small(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False
# แช่แข็ง BatchNorm ทั้งหมด (batch เล็ก + ข้อมูลน้อย ทำให้สถิติ BN เพี้ยน)
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])

os.makedirs("weights", exist_ok=True)
with open("weights/class_mapping.json", "w", encoding="utf-8") as f:
    json.dump({str(v): k for k, v in train_generator.class_indices.items()}, f, indent=4)

checkpoint_path = "weights/best.keras"
if has_val:
    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ]
    model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS,
              callbacks=callbacks, class_weight=class_weights)
else:
    print(f"ไม่มี val: เทรน {FINAL_EPOCHS} epoch ด้วยข้อมูลทั้งหมด")
    model.fit(train_generator, epochs=FINAL_EPOCHS, class_weight=class_weights)
    model.save(checkpoint_path)

print(f"บันทึกโมเดลไว้ที่: {checkpoint_path}")
