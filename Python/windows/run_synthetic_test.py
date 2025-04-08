import os
import numpy as np
import tensorflow as tf
import librosa
from datetime import datetime
from configs.settings import OUTPUT_CLASSES, SAMPLE_RATE, CLASS_MAPPING, INPUT_SHAPE
from evaluation.confusion_matrix import plot_and_save_confusion_matrix

# ตั้งค่า logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

def load_and_process_audio(file_path):
    """โหลดและประมวลผลไฟล์เสียง"""
    try:
        if file_path.endswith('.wav'):
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            return y
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def extract_features(audio):
    """สกัดคุณลักษณะจากเสียง"""
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=13,
        n_fft=2048,
        hop_length=512
    )
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    flattened = mfcc.T.flatten()
    return np.pad(flattened, (0, INPUT_SHAPE[1] - len(flattened)))[:INPUT_SHAPE[1]]

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
    confidences = []

    print("\n=== ทดสอบกับเสียงจริง ===")
    audio_files = [f for f in os.listdir("real_audio_samples") if f.endswith('.wav')]
    
    for audio_file in audio_files[:20]:  # ลองกับ 20 ไฟล์แรกเพื่อทดสอบ
        try:
            # ดึงโน้ตจากชื่อไฟล์ เช่น A.A_min_7_0.wav.xxx.wav → 'A'
            base_name = audio_file.split('.')[0]
            note = base_name.upper()
            if note not in OUTPUT_CLASSES:
                continue

            audio_path = os.path.join("real_audio_samples", audio_file)
            audio = load_and_process_audio(audio_path)

            if audio is None:
                continue

            features = extract_features(audio)

            # Quantize input
            quant_input = np.clip(
                (features / input_scale + input_zero_point),
                -128, 127
            ).astype(np.int8)

            # รันโมเดล
            interpreter.set_tensor(input_details[0]['index'], np.expand_dims(quant_input, axis=0))
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])

            # Dequantize
            dequant_output = (output.astype(np.float32) - output_zero_point) * output_scale
            probabilities = tf.nn.softmax(dequant_output).numpy()[0]
            predicted_index = np.argmax(probabilities)

            y_true.append(note)
            y_pred.append(OUTPUT_CLASSES[predicted_index])
            confidences.append(probabilities[predicted_index])

            print(f"\nไฟล์: {audio_file}")
            print(f"โน้ตจริง: {note} | ทำนาย: {OUTPUT_CLASSES[predicted_index]} ({probabilities[predicted_index]:.2%})")
            print("ความน่าจะเป็นทั้งหมด:")
            for i, p in enumerate(probabilities):
                print(f"{CLASS_MAPPING[i]}: {p:.2%}")

        except Exception as e:
            print(f"Error processing file {audio_file}: {str(e)}")

    # สร้าง Confusion Matrix
    if y_true:
        plot_and_save_confusion_matrix(y_true, y_pred, "confusion_matrix_real.png")
        
        # สร้างรายงาน
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        print(f"\nความแม่นยำ: {accuracy:.2%}")
    else:
        print("\nไม่พบไฟล์เสียงที่สามารถทดสอบได้")

if __name__ == "__main__":
    main()