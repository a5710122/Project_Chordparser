# การตั้งค่าเสียง
SAMPLE_RATE = 16000  # Hz
AUDIO_LENGTH_SEC = 1.0  # ความยาวเสียง (วินาที)

# การตั้งค่าโมเดล
INPUT_SHAPE = (1, 6435)  # ต้องตรงกับโมเดลของคุณ

# การแมปคลาส
OUTPUT_CLASSES = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
CLASS_MAPPING = {
    0: 'C',
    1: 'D',
    2: 'E',
    3: 'F',
    4: 'G',
    5: 'A',
    6: 'B'
}