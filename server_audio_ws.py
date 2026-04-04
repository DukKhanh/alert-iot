# server.py
import socket
import numpy as np
import librosa
import time
from collections import deque
from tensorflow.keras.models import load_model
from flask import Flask
from flask_socketio import SocketIO
import threading

# =========================
# CONFIG
# =========================
UDP_IP = "0.0.0.0"
UDP_PORT = 8888

ESP32_PORT = 9999

SAMPLE_RATE = 16000
BUFFER_SECONDS = 1
BUFFER_SIZE = SAMPLE_RATE * BUFFER_SECONDS

PREDICT_INTERVAL = 1.0
ALERT_THRESHOLD = 0.5
ALERT_COOLDOWN = 2.0
HISTORY_SIZE = 5

# =========================
# LABELS
# =========================
LABELS = {
    0: "Background",
    1: "Glass Breaking",
    2: "Gunshot",
    3: "Scream"
}

# =========================
# INIT SERVER
# =========================
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Shared state cho latest spectrogram / waveform
latest_spectrogram = {"data": [], "shape": [0, 0], "db_min": -80.0, "db_max": 0.0}
latest_waveform = np.array([], dtype=np.float32)

@app.route("/")
def home():
    return "Server OK"

@app.route("/api/spectrogram")
def get_spectrogram():
    """REST endpoint: trả về mel-spectrogram mới nhất dạng JSON."""
    from flask import jsonify
    return jsonify(latest_spectrogram)

@app.route("/api/waveform")
def get_waveform():
    """REST endpoint: trả về waveform mới nhất."""
    from flask import jsonify
    return jsonify({"samples": latest_waveform.tolist() if len(latest_waveform) > 0 else []})

@socketio.on('connect')
def connect():
    print("Client connected!")
    # Gửi ngay frame hiện tại cho client mới kết nối
    if len(latest_spectrogram["data"]) > 0:
        socketio.emit('spectrogram', latest_spectrogram)

# =========================
# LOAD MODEL
# =========================
print("Loading model...")
model = load_model("spectrogram_model_best.keras", compile=False)
print("Model loaded!")

# =========================
# PROCESSOR
# =========================
class MIVIASpectrogramProcessor:
    def __init__(self, sample_rate=16000, frame_size_ms=50, seq_length=10,
                 n_mels=128, n_fft=1024, hop_length=256):
        self.sample_rate = sample_rate
        self.frame_size_ms = frame_size_ms
        self.seq_length = seq_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.frame_samples = int(self.sample_rate * self.frame_size_ms / 1000)
        self.n_time_frames = int(np.ceil(self.frame_samples / self.hop_length))

    def _audio_to_mel_frames(self, audio):
        if len(audio) == 0:
            return None
        D = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length,
                         window='hann', center=False)
        mel = librosa.feature.melspectrogram(S=np.abs(D) ** 2,
                                             sr=self.sample_rate,
                                             n_mels=self.n_mels)
        log_mel = librosa.power_to_db(mel, ref=1.0)
        total_time_frames = log_mel.shape[1]
        step = max(1, int(self.n_time_frames * 0.75))
        if total_time_frames < self.n_time_frames:
            pad_width = self.n_time_frames - total_time_frames
        else:
            remainder = (total_time_frames - self.n_time_frames) % step
            pad_width = (step - remainder) if remainder != 0 else 0
        if pad_width > 0:
            log_mel = np.pad(log_mel,
                             ((0, 0), (0, pad_width)),
                             mode='constant',
                             constant_values=log_mel.min())
        frames = []
        start = 0
        total_time_frames = log_mel.shape[1]
        while start + self.n_time_frames <= total_time_frames:
            frame = log_mel[:, start:start + self.n_time_frames]
            frames.append(frame)
            start += step
        return np.array(frames, dtype=np.float32)

    def extract_features(self, audio):
        frames = self._audio_to_mel_frames(audio)
        if frames is None or len(frames) == 0:
            return None
        num_frames = frames.shape[0]
        remainder = num_frames % self.seq_length
        if remainder != 0:
            pad_width = self.seq_length - remainder
            frames = np.pad(frames,
                            ((0, pad_width), (0, 0), (0, 0)),
                            mode='constant',
                            constant_values=frames.min())
        num_sequences = frames.shape[0] // self.seq_length
        sequences = frames.reshape((num_sequences, self.seq_length,
                                    self.n_mels, self.n_time_frames))
        return sequences

# =========================
# HELPER
# =========================
def get_class_id(label):
    for k, v in LABELS.items():
        if v == label:
            return k
    return 0

# =========================
# MODEL PREDICT
# =========================
def model_predict(X):
    pred = model.predict(X, verbose=0)
    if len(pred.shape) == 3:
        avg_pred = np.mean(pred, axis=(0, 1))
    else:
        avg_pred = np.mean(pred, axis=0)
    label_id = int(np.argmax(avg_pred))
    prob = float(avg_pred[label_id])
    return prob, LABELS[label_id], avg_pred

# =========================
# UDP LOOP
# =========================
def compute_realtime_spectrogram(audio_np, sample_rate=16000, n_mels=128, n_fft=1024, hop_length=256):
    """Tính mel-spectrogram từ audio buffer, trả về dict sẵn sàng emit."""
    if len(audio_np) < n_fft:
        return None
    D = librosa.stft(audio_np, n_fft=n_fft, hop_length=hop_length, window='hann', center=False)
    mel = librosa.feature.melspectrogram(S=np.abs(D) ** 2, sr=sample_rate, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=1.0)
    db_min = float(log_mel.min())
    db_max = float(log_mel.max())
    # Normalize về [0, 1] để FE dễ render
    if db_max > db_min:
        norm = ((log_mel - db_min) / (db_max - db_min)).astype(np.float32)
    else:
        norm = np.zeros_like(log_mel, dtype=np.float32)
    return {
        "data": norm.tolist(),          # shape [n_mels, time_frames]
        "shape": list(norm.shape),
        "db_min": round(db_min, 2),
        "db_max": round(db_max, 2)
    }


def udp_loop():
    global latest_spectrogram, latest_waveform

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print("Listening UDP...")

    audio_buffer = deque(maxlen=BUFFER_SIZE)
    processor = MIVIASpectrogramProcessor()

    last_predict_time = 0
    last_alert_time = 0
    last_spec_time = 0
    pred_history = []

    # Thêm biến này để tự động lấy IP của ESP32
    current_esp32_ip = None
    while True:
        data, addr = sock.recvfrom(4096)
        
        # BƯỚC MỚI: Xử lý dò tìm IP (Handshake)
        if data == b"DISCOVER":
            print(f"[*] Nhận yêu cầu kết nối từ ESP32 tại IP: {addr[0]}")
            sock.sendto(b"SERVER_HERE", (addr[0], ESP32_PORT))
            continue # Bỏ qua vòng lặp này vì đây không phải là data âm thanh

        # THÊM 1 DÒNG NÀY ĐỂ LƯU IP ESP32:
        current_esp32_ip = addr[0]
        audio_chunk = np.frombuffer(data, dtype=np.int32)
        # audio_chunk = audio_chunk >> 8
        audio_chunk = audio_chunk.astype(np.float32) / (2**23)
        audio_buffer.extend(audio_chunk)

        # ===== SEND WAVEFORM TO UI =====
        wave_frame = np.array(audio_buffer)[-1024:]  # 1024 mẫu cuối
        latest_waveform = wave_frame
        socketio.emit('waveform', {'samples': wave_frame.tolist()})

        # ===== SEND SPECTROGRAM REALTIME (mỗi 200ms) =====
        current_time = time.time()
        if current_time - last_spec_time >= 0.2 and len(audio_buffer) >= 1024:
            last_spec_time = current_time
            spec_audio = np.array(audio_buffer)[-SAMPLE_RATE:]  # 1 giây gần nhất
            spec_data = compute_realtime_spectrogram(spec_audio)
            if spec_data:
                latest_spectrogram = spec_data
                socketio.emit('spectrogram', spec_data)

        current_time = time.time()
        if len(audio_buffer) >= BUFFER_SIZE and (current_time - last_predict_time > PREDICT_INTERVAL):
            last_predict_time = current_time
            audio_np = np.array(audio_buffer)
            X = processor.extract_features(audio_np)
            if X is None:
                continue
            prob, label, full_probs = model_predict(X)
            pred_history.append(prob)
            if len(pred_history) > HISTORY_SIZE:
                pred_history.pop(0)
            avg_prob = float(np.mean(pred_history))
            class_id = get_class_id(label)

            # ===== SEND PREDICTION TO UI =====
            socketio.emit('audio_event', {
                "label": label,
                "class_id": class_id,
                "confidence": avg_prob,
                "confs": full_probs.tolist()
            })

            # ===== ALERT TO ESP32 =====
            current_state = 1 if (label != "Background" and avg_prob > ALERT_THRESHOLD) else 0
            if current_esp32_ip is not None:
                if current_state == 1 and (current_time - last_alert_time > ALERT_COOLDOWN):
                    print("\n🚨 ALERT:", label.upper(), "Confidence:", round(avg_prob, 3))
                    sock.sendto(bytes([1]), (current_esp32_ip, ESP32_PORT))
                    last_alert_time = current_time
                elif current_state == 0:
                    sock.sendto(bytes([0]), (current_esp32_ip, ESP32_PORT))

# =========================
# START SERVER
# =========================
if __name__ == "__main__":
    threading.Thread(target=udp_loop, daemon=True).start()
    socketio.run(app, host="0.0.0.0", port=5000)