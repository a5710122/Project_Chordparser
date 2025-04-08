import librosa
import numpy as np
from configs.settings import SAMPLE_RATE, INPUT_SHAPE

def extract_mfcc(audio_path: str) -> np.ndarray:
    """
    สกัด MFCC features จากไฟล์เสียง .wav
    ค่าพารามิเตอร์ต้องตรงกับที่ใช้ฝึกโมเดลใน Edge Impulse!
    """
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    
    # ค่าเหล่านี้ต้องตรงกับที่ใช้ใน Edge Impulse
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40,       # จำนวน MFCC coefficients
        n_fft=1024,      # ขนาดหน้าต่าง FFT
        hop_length=512,   # การเลื่อนหน้าต่าง
        win_length=1024,  # ความยาวหน้าต่าง
        window='hann'     # ชนิดหน้าต่าง
    )
    
    # เพิ่ม Delta และ Delta-Delta (ถ้าใช้ใน Edge Impulse)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    # รวม features
    features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
    flattened = features.T.flatten()
    
    return _adjust_length(flattened, INPUT_SHAPE[1])

def extract_features_from_array(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    สกัด features จาก audio array (สำหรับเสียงสังเคราะห์)
    ใช้พารามิเตอร์เดียวกับ extract_mfcc
    """
    # ตรวจสอบและปรับขนาดเสียงให้เหมาะสม
    if len(audio) < 1024:
        audio = np.pad(audio, (0, 1024 - len(audio)))
    
    # คำนวณ MFCC (ใช้พารามิเตอร์เหมือนด้านบน)
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40,
        n_fft=1024,
        hop_length=512,
        win_length=1024,
        window='hann'
    )
    
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
    flattened = features.T.flatten()
    
    return _adjust_length(flattened, INPUT_SHAPE[1])

def _adjust_length(features: np.ndarray, target_len: int) -> np.ndarray:
    """ปรับขนาด features ให้ตรงกับ input shape ของโมเดล"""
    if len(features) > target_len:
        return features[:target_len]
    return np.pad(features, (0, target_len - len(features)))

# ฟังก์ชันสำหรับตรวจสอบ features
def print_feature_stats(features: np.ndarray):
    """พิมพ์สถิติของ features สำหรับ debugging"""
    print("\n=== Feature Statistics ===")
    print(f"Shape: {features.shape}")
    print(f"Min: {np.min(features):.4f}")
    print(f"Max: {np.max(features):.4f}")
    print(f"Mean: {np.mean(features):.4f}")
    print(f"Std: {np.std(features):.4f}")
    print("ตัวอย่างค่า:", features[:5])

def load_real_audio(file_path: str, target_length: int = 16000):
    """โหลดเสียงจริงจากไฟล์และปรับความยาว"""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]
    return y