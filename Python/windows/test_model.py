import os
import numpy as np
import tensorflow as tf
from datetime import datetime

# ฟังก์ชันสำหรับบันทึกผลลัพธ์
def save_to_txt(content, filename="model_test_log.txt"):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(content + "\n")

# ระบบ Mapping สำหรับคลาส (ปรับตามโมเดลของคุณ)
CLASS_MAPPING = {
    0: "C",
    1: "D", 
    2: "E",
    3: "F",
    4: "G",
    5: "A",
    6: "B"
}

# ปิด warning oneDNN
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

# บันทึกข้อมูลโมเดล (เพิ่ม quantization details)
model_info = f"""
=== ข้อมูลโมเดล ===
Input Shape: {input_details[0]['shape']}
Input Type: {input_details[0]['dtype']}
Input Quantization: {input_details[0]['quantization']}

Output Shape: {output_details[0]['shape']}
Output Type: {output_details[0]['dtype']}
Output Quantization: {output_details[0]['quantization']}

Class Mapping: {CLASS_MAPPING}
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

# บันทึกผลลัพธ์ดิบ
save_to_txt("\n=== ผลลัพธ์ทดสอบ ===")
save_to_txt(f"ผลลัพธ์ดิบ (int8):\n{str(output_data)}")

# แปลงผลลัพธ์และแสดง Mapping
if output_details[0]['quantization'] != (0.0, 0):
    scale, zero_point = output_details[0]['quantization']
    save_to_txt(f"\n=== ขั้นตอน Dequantization ===")
    save_to_txt(f"สูตร: (output - {zero_point}) * {scale}")
    
    # แสดงการคำนวณแบบทีละขั้นตอน
    save_to_txt("\nการคำนวณแต่ละค่า:")
    for i, val in enumerate(output_data[0]):
        dequant_val = (val - zero_point) * scale
        save_to_txt(f"Index {i} ({CLASS_MAPPING.get(i, 'Unknown')}): {val} -> ({val} - {zero_point}) * {scale} = {dequant_val:.6f}")
    
    # คำนวณทั้งหมด
    dequantized_output = scale * (output_data.astype(np.float32) - zero_point)
    save_to_txt(f"\nผลลัพธ์หลัง Dequantize (float32):\n{str(dequantized_output)}")
    
    # แปลงเป็นความน่าจะเป็นและแสดงผลลัพธ์
    probabilities = tf.nn.softmax(dequantized_output).numpy()[0]
    predicted_index = np.argmax(probabilities)
    
    save_to_txt("\n=== ผลลัพธ์การทำนาย ===")
    save_to_txt(f"โน้ตที่ทำนาย: {CLASS_MAPPING.get(predicted_index, 'Unknown')}")
    save_to_txt(f"ความมั่นใจ: {probabilities[predicted_index]:.2%}")
    
    save_to_txt("\nความน่าจะเป็นของทุกคลาส:")
    for i, prob in enumerate(probabilities):
        save_to_txt(f"{CLASS_MAPPING.get(i, 'Unknown')}: {prob:.2%}")

save_to_txt("\n===== สิ้นสุดการทดสอบ =====\n")
print("บันทึกผลลัพธ์เรียบร้อยที่ไฟล์ 'model_test_log.txt'")