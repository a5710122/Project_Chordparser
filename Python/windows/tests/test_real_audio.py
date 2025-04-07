import tensorflow as tf
from audio_utils.feature_extraction import extract_mfcc
from configs.settings import OUTPUT_CLASSES

def test_single_audio(model_path: str, audio_path: str) -> dict:
    """ทดสอบโมเดลกับไฟล์เสียงเดียว"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    features = extract_mfcc(audio_path)
    
    # เตรียม input (ต้องตรงกับโมเดล)
    input_data = (features * 127).astype(np.int8)  # ตัวอย่างสำหรับโมเดล int8
    input_data = np.expand_dims(input_data, axis=0)
    
    # รันโมเดล
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # ประมวลผลผลลัพธ์
    output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    return {OUTPUT_CLASSES[i]: float(output[0][i]) for i in range(len(OUTPUT_CLASSES))}