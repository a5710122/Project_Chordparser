import os
import numpy as np
import tensorflow as tf
from datetime import datetime

# ฟังก์ชันสำหรับบันทึกผลลัพธ์
def save_to_txt(content, filename="model_test_log.txt"):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(content + "\n")

# ปิด warning oneDNN (Optional)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# กำหนด path โมเดล
model_path = os.path.join(os.getcwd(), "models", "trained.tflite")

# สร้างหัวข้อในไฟล์บันทึก
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
save_to_txt(f"\n===== การทดสอบโมเดล [{timestamp}] =====")

try:
    # โหลดโมเดล
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    save_to_txt("✅ โหลดโมเดลสำเร็จ!")
except Exception as e:
    error_msg = f"❌ เกิดข้อผิดพลาดขณะโหลดโมเดล: {str(e)}"
    save_to_txt(error_msg)
    print(error_msg)
    exit()

# ดูรายละเอียด Input/Output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# บันทึกข้อมูลโมเดล
model_info = f"""
=== ข้อมูลโมเดล ===
Input Shape: {input_details[0]['shape']}
Input Type: {input_details[0]['dtype']}
Output Shape: {output_details[0]['shape']}
Output Type: {output_details[0]['dtype']}
"""
save_to_txt(model_info)

# สร้างข้อมูลทดสอบ
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

# บันทึกผลลัพธ์
save_to_txt("\n=== ผลลัพธ์ทดสอบ ===")
save_to_txt(f"ผลลัพธ์ดิบ (int8):\n{str(output_data)}")

# ถ้าโมเดลใช้ Quantization
if output_details[0]['quantization'] != (0.0, 0):
    scale, zero_point = output_details[0]['quantization']
    dequantized_output = scale * (output_data.astype(np.float32) - zero_point)
    save_to_txt(f"\nผลลัพธ์หลัง Dequantize (float32):\n{str(dequantized_output)}")

save_to_txt("\n===== สิ้นสุดการทดสอบ =====\n")
print("บันทึกผลลัพธ์เรียบร้อยที่ไฟล์ 'model_test_log.txt'")