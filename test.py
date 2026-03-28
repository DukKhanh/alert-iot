from predict import predict_single_audio
from processor import MIVIASpectrogramProcessor
import tensorflow as tf
from tensorflow.keras.models import load_model

SAMPLE_RATE = 16000
SEQ_LENGTH = 10
N_MELS = 128

# Khởi tạo Processor với đúng các tham số bạn đã dùng khi train
processor = MIVIASpectrogramProcessor(
    sample_rate=SAMPLE_RATE, 
    frame_size_ms=50, 
    seq_length=SEQ_LENGTH,
    n_mels=N_MELS, 
    n_fft=1024, 
    hop_length=256
)

# Load model đã train từ file .keras
print("Loading model 'spectrogram_model_best.keras'...")
model = load_model('spectrogram_model_best.keras')

# Chỉ định đường dẫn tới 1 file audio bất kỳ bạn muốn test
# (Thay đổi đường dẫn này trỏ tới file wav bạn muốn kiểm tra)
sample_audio_file = "00001_00.wav"

# Chạy Inference
detected_events = predict_single_audio(
    audio_path=sample_audio_file, 
    model=model, 
    processor=processor,
    sample_rate=SAMPLE_RATE
)

# In kết quả
print("\n=== KẾT QUẢ NHẬN DIỆN TỪ MODEL ===")
if len(detected_events) == 0:
    print("✅ Chỉ phát hiện Background (Không có sự kiện bất thường).")
else:
    for ev in detected_events:
        print(f"🚨 Phát hiện: {ev['event']} | Từ: {ev['start_sec']}s đến {ev['end_sec']}s (Kéo dài {ev['duration']}s)")