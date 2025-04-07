import os
import numpy as np
import tensorflow as tf
from audio_utils.audio_generator import AudioGenerator
from audio_utils.feature_extraction import extract_features_from_array
from configs.settings import OUTPUT_CLASSES, SAMPLE_RATE
import matplotlib.pyplot as plt

def main():
    # 1. โหลดโมเดล
    model_path = os.path.abspath("models/trained.tflite")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดลที่ {model_path}")
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 2. ตรวจสอบ Quantization
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    print("\n=== ข้อมูลโมเดล ===")
    print(f"Input Scale: {input_scale}, Zero Point: {input_zero_point}")
    print(f"Output Scale: {output_scale}, Zero Point: {output_zero_point}")

    # 3. ทดสอบกับโน้ตทุกตัว
    y_true = []
    y_pred = []
    confidences = []

    print("\n=== ผลลัพธ์การทำนาย ===")
    for note in OUTPUT_CLASSES:
        # สร้างเสียงและสกัดคุณลักษณะ
        audio = AudioGenerator.generate_note(note)
        features = extract_features_from_array(audio, SAMPLE_RATE)
        
        # Quantize ข้อมูล Input
        quantized_input = (features / input_scale + input_zero_point).astype(np.int8)
        input_data = np.expand_dims(quantized_input, axis=0)
        
        # รันโมเดล
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Dequantize ผลลัพธ์
        output = interpreter.get_tensor(output_details[0]['index'])
        dequantized_output = output_scale * (output.astype(np.float32) - output_zero_point)
        
        # คำนวณความน่าจะเป็น
        probabilities = tf.nn.softmax(dequantized_output).numpy()[0]
        predicted_note = OUTPUT_CLASSES[np.argmax(probabilities)]
        confidence = np.max(probabilities)
        
        y_true.append(note)
        y_pred.append(predicted_note)
        confidences.append(confidence)
        
        print(f"โน้ตจริง: {note} | ทำนาย: {predicted_note} ({confidence:.2%})")

    # 4. บันทึกผลลัพธ์ (แก้ไข encoding)
    with open("synthetic_test_results.txt", "w", encoding="utf-8") as f:
        f.write("=== ผลลัพธ์การทดสอบ ===\n")
        for true, pred, conf in zip(y_true, y_pred, confidences):
            f.write(f"{true} -> {pred} ({conf:.2%})\n")

if __name__ == "__main__":
    main()