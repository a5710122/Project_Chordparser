import numpy as np
import tensorflow as tf
from audio_utils.audio_generator import AudioGenerator
from audio_utils.feature_extraction import extract_features_from_array
from configs.settings import OUTPUT_CLASSES

def test_synthetic_notes(model_path: str) -> dict:
    """ทดสอบโมเดลกับเสียงสังเคราะห์ทั้งหมด"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    
    results = {}
    
    for note in OUTPUT_CLASSES:
        # สร้างเสียงโน้ต
        audio = AudioGenerator.generate_note(note)
        
        # สกัดคุณลักษณะ (ใช้ฟังก์ชันที่รับ array โดยตรง)
        features = extract_features_from_array(audio, SAMPLE_RATE)
        
        # เตรียม input
        input_data = (features * 127).astype(np.int8)  # สำหรับโมเดล int8
        input_data = np.expand_dims(input_data, axis=0)
        
        # รันโมเดล
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # เก็บผลลัพธ์
        output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        results[note] = {
            'raw_output': output[0],
            'predicted': OUTPUT_CLASSES[np.argmax(output[0])]
        }
    
    return results

# เพิ่มฟังก์ชันใน audio_utils/feature_extraction.py
def extract_features_from_array(audio: np.ndarray, sr: int) -> np.ndarray:
    """สกัด features จาก audio array"""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    flattened = mfcc.T.flatten()
    return _adjust_length(flattened, INPUT_SHAPE[1])