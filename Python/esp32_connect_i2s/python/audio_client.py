import time
import socket
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.io import wavfile

# แก้ตรงนี้!
SERVER_IP = '192.168.1.43'
SERVER_PORT = 9000
BUF_LEN     = 512
SAMPLE_RATE = 16000
DURATION    = 5  # seconds

def wait_for_ready_signal(sock):
    print("📡 Waiting for ESP32...")
    while True:
        data = sock.recv(1024).decode('utf-8')
        if 'ready' in data:
            print("✅ ESP32 is ready")
            break

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((SERVER_IP, SERVER_PORT))
wait_for_ready_signal(sock)

def read_audio_data(duration=DURATION, fname="audio.wav"):
    num_samples = SAMPLE_RATE * duration
    audio_data = np.zeros(num_samples, dtype=np.int16)
    sample_idx = 0

    print("🎙️ Receiving audio...")
    end_time = time.time() + duration

    while time.time() < end_time:
        data = sock.recv(BUF_LEN * 2)
        if data:
            samples = np.frombuffer(data, dtype=np.int16)
            if sample_idx == 0:
                samples = samples[2:]  # ข้าม sample แรก
            end_idx = sample_idx + len(samples)
            if end_idx > num_samples:
                end_idx = num_samples
                samples = samples[:end_idx - sample_idx]
            audio_data[sample_idx:end_idx] = samples
            sample_idx = end_idx

    wavfile.write(fname, SAMPLE_RATE, audio_data)
    print(f"💾 Saved to: {fname}")
    return audio_data

def plot_audio_data(audio_data):
    time_axis = np.linspace(0, len(audio_data) / SAMPLE_RATE, num=len(audio_data))
    plt.figure(figsize=(10, 8))

    # Waveform
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, audio_data)
    plt.title('🟦 Time Series (Waveform)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    # Spectrogram
    plt.subplot(2, 1, 2)
    f, t, Sxx = spectrogram(audio_data, SAMPLE_RATE)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-8), shading='gouraud')
    plt.title('🌈 Spectrogram')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar(label='Intensity [dB]')
    plt.tight_layout()
    plt.show()

audio_data = read_audio_data()
sock.close()
plot_audio_data(audio_data)
