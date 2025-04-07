import os
import numpy as np
import tensorflow as tf

# ปิด warning oneDNN (Optional)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# กำหนด path โมเดล (ใช้ path แบบเต็มเพื่อความชัดเจน)
model_path = os.path.join(os.getcwd(), "model", "trained.tflite")

# โหลดโมเดล
try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print("โหลดโมเดลสำเร็จ! ✅")
except Exception as e:
    print(f"เกิดข้อผิดพลาดขณะโหลดโมเดล: {e}")
    exit()

# ดูรายละเอียด Input/Output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\n=== ข้อมูลโมเดล ===")
print(f"Input Shape: {input_details[0]['shape']}")  # ควรเป็น [1, 6435]
print(f"Input Type: {input_details[0]['dtype']}")   # ควรเป็น int8
print(f"\nOutput Shape: {output_details[0]['shape']}")  # ควรเป็น [1, 7]
print(f"Output Type: {output_details[0]['dtype']}")     # ควรเป็น int8

# สร้างข้อมูลทดสอบ (ปรับตามโมเดลของคุณ)
input_data = np.random.randint(
    low=-128, 
    high=127, 
    size=input_details[0]['shape'], 
    dtype=np.int8
)

# ใส่ข้อมูลและรันโมเดล
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# รับผลลัพธ์
output_data = interpreter.get_tensor(output_details[0]['index'])
print("\n=== ผลลัพธ์ทดสอบ ===")
print("ผลลัพธ์ดิบ (int8):\n", output_data)

# ถ้าโมเดลใช้ Quantization
if output_details[0]['quantization'] != (0.0, 0):
    scale, zero_point = output_details[0]['quantization']
    dequantized_output = scale * (output_data.astype(np.float32) - zero_point)
    print("\nผลลัพธ์หลัง Dequantize (float32):\n", dequantized_output)