import numpy as np
from configs.settings import SAMPLE_RATE, AUDIO_LENGTH_SEC

class AudioGenerator:
    """คลาสสำหรับสร้างเสียงสังเคราะห์เพื่อทดสอบโมเดล"""
    
    @staticmethod
    def generate_sine_wave(freq: float, amplitude: float = 0.5) -> np.ndarray:
        """สร้างเสียงไซน์เวฟจากความถี่ที่กำหนด"""
        t = np.linspace(0, AUDIO_LENGTH_SEC, int(SAMPLE_RATE * AUDIO_LENGTH_SEC))
        return amplitude * np.sin(2 * np.pi * freq * t)
    
    @staticmethod
    def generate_note(note_name: str) -> np.ndarray:
        """สร้างเสียงโน้ตดนตรีจากชื่อโน้ต"""
        note_frequencies = {
            'C': 261.63,  # C4
            'D': 293.66,
            'E': 329.63,
            'F': 349.23,
            'G': 392.00,
            'A': 440.00,
            'B': 493.88
        }
        return AudioGenerator.generate_sine_wave(note_frequencies[note_name])
    
    @staticmethod
    def add_noise(audio: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """เพิ่มสัญญาณรบกวนแบบ Gaussian"""
        noise = np.random.normal(0, noise_level, len(audio))
        return audio + noise