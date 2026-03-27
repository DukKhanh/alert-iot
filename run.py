import socket
import numpy as np
import tensorflow as tf

from custom_layers import SequenceToBatch, BatchToSequence, SincConv1D

# ===== CONFIG =====
ESP_IP = "10.223.17.147"
PORT = 5000

FRAME_LENGTH = 800
SEQ_LENGTH = 10
RECV_SIZE = 800 * 2  # int16

AMP_THRESHOLD = 1000   # 🔥 thêm
CONF_THRESHOLD = 0.6   # 🔥 sửa từ 0.7 → 0.6

# ===== LOAD MODEL =====
model = tf.keras.models.load_model(
    "Tiny_SincNet_best.keras",
    compile=False,
    custom_objects={
        "SequenceToBatch": SequenceToBatch,
        "BatchToSequence": BatchToSequence,
        "SincConv1D": SincConv1D
    }
)
print("✅ Model loaded")

# ===== SOCKET =====
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((ESP_IP, PORT))
print("✅ Connected to ESP32")

# ===== RECEIVE EXACT =====
def recv_exact(sock, size):
    data = b''
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data

frames = []
history = []   # 🔥 thêm smooth

while True:
    data = recv_exact(sock, RECV_SIZE)

    if data is None:
        continue

    # ===== CONVERT =====
    frame = np.frombuffer(data, dtype=np.int16)

    if frame.shape[0] != FRAME_LENGTH:   # 🔥 thêm check
        continue

    # DEBUG
    print("MIN:", np.min(frame), "MAX:", np.max(frame))

    # ===== AMP CHECK ===== 🔥 thêm
    amp = np.max(np.abs(frame))

    # ===== NORMALIZE ===== 🔥 cải thiện
    frame = frame.astype(np.float32) / 32768.0
    frame = frame - np.mean(frame)

    frames.append(frame)

    # ===== ENOUGH FRAMES =====
    if len(frames) == SEQ_LENGTH:

        segment = np.array(frames).reshape(1, SEQ_LENGTH, FRAME_LENGTH)

        # ===== AI =====
        pred = model.predict(segment, verbose=0)

        # ===== FIX SHAPE 🔥 quan trọng
        if len(pred.shape) == 3:
            pred = np.mean(pred, axis=1)[0]
        else:
            pred = pred[0]

        label = int(np.argmax(pred))
        confidence = float(np.max(pred))

        labels = ["Background", "Glass", "Gunshot", "Scream"]
        print("Raw:", labels[label], "Conf:", confidence, "AMP:", amp)

        # ===== FILTER 🔥 quan trọng nhất
        if amp < AMP_THRESHOLD:
            final_label = 0
            print("FORCED Background")
        else:
            if confidence > CONF_THRESHOLD:
                final_label = label
            else:
                final_label = 0

        # ===== SMOOTH 🔥 chống nhảy
        history.append(final_label)
        if len(history) > 5:
            history.pop(0)

        final_label = max(set(history), key=history.count)

        print("FINAL:", labels[final_label])
        print("------------------------")

        # ===== SEND BACK =====
        sock.send(bytes([final_label]))

        frames = []