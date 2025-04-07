import os
import numpy as np
import tensorflow as tf
from audio_utils.audio_generator import AudioGenerator
from audio_utils.feature_extraction import extract_features_from_array
from configs.settings import OUTPUT_CLASSES, SAMPLE_RATE
from evaluation.confusion_matrix import plot_and_save_confusion_matrix

# ปิด warning oneDNN (Optional)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def main():
    # โหลดโมเดล
    model_path = os.path.abspath("models/trained.tflite")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # ข้อมูล Quantization
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']
    
    y_true = []
    y_pred = []
    
    print("\n=== ผลลัพธ์การทำนาย ===")
    for note in OUTPUT_CLASSES:
        # สร้างเสียงและสกัดคุณลักษณะ
        audio = AudioGenerator.generate_note(note)
        features = extract_features_from_array(audio, SAMPLE_RATE)
        
        # Quantize ข้อมูล
        quantized_input = (features / input_scale + input_zero_point).astype(np.int8)
        input_data = np.expand_dims(quantized_input, axis=0)
        
        # รันโมเดล
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # ประมวลผลผลลัพธ์
        output = interpreter.get_tensor(output_details[0]['index'])
        dequantized = (output.astype(np.float32) - output_zero_point) * output_scale
        probabilities = tf.nn.softmax(dequantized).numpy()[0]
        
        predicted_note = OUTPUT_CLASSES[np.argmax(probabilities)]
        confidence = np.max(probabilities)
        
        y_true.append(note)
        y_pred.append(predicted_note)
        
        print(f"โน้ตจริง: {note} | ทำนาย: {predicted_note} ({confidence:.2%})")
    
    # สร้างและบันทึก Confusion Matrix
    plot_and_save_confusion_matrix(y_true, y_pred)

if __name__ == "__main__":
    main()