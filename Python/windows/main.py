from tests.test_real_audio import test_single_audio

MODEL_PATH = "models/trained.tflite"
AUDIO_PATH = "test_samples/C_note.wav"

if __name__ == "__main__":
    results = test_single_audio(MODEL_PATH, AUDIO_PATH)
    print("ผลลัพธ์การทำนาย:", results)