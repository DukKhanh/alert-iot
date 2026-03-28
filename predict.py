import librosa
import numpy as np
from scipy.signal import medfilt
from scipy.ndimage import uniform_filter1d

def predict_single_audio(audio_path, model, processor, sample_rate=16000, confidence_threshold=0.75, min_duration=0.3):
    """
    Chạy inference trên 1 file audio: Tích hợp Smooth Probability, Confidence Threshold và Merge Events.
    """
    print(f"Loading {audio_path}...")
    
    try:
        audio_data, sr = librosa.load(audio_path, sr=sample_rate)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return[]

    # Tạo dummy data để dùng lại processor (Label = 0 chỉ là label giả)
    dummy_data = [{'audio': audio_data, 'label': 0}]
    X_infer, _ = processor.extract_features(dummy_data, verbose=False)
    
    if len(X_infer) == 0:
        print("Audio is too short or empty!")
        return[]

    # --- BƯỚC 1: DỰ ĐOÁN VÀ LÀM MƯỢT XÁC SUẤT ---
    y_pred_probs = model.predict(X_infer, verbose=0)
    num_classes = y_pred_probs.shape[-1]
    y_pred_probs_flat = y_pred_probs.reshape(-1, num_classes)
    
    # Trượt trung bình trên 5 frames (~0.24 giây) để dập tắt nhiễu chớp nhoáng
    y_pred_probs_smoothed = uniform_filter1d(y_pred_probs_flat, size=5, axis=0)

    # --- BƯỚC 2: ÁP DỤNG NGƯỠNG TỰ TIN (CONFIDENCE THRESHOLD) ---
    y_pred_raw = np.argmax(y_pred_probs_smoothed, axis=-1)
    max_probs = np.max(y_pred_probs_smoothed, axis=-1)
    
    # Ép về Background (0) nếu xác suất cao nhất không đạt ngưỡng (vd: < 75%)
    y_pred_raw[max_probs < confidence_threshold] = 0

    # --- BƯỚC 3: BỘ LỌC MEDIAN (MEDIAN FILTER) ---
    if len(y_pred_raw) >= 7:
        y_pred_smoothed = medfilt(y_pred_raw, kernel_size=7)
    else:
        y_pred_smoothed = y_pred_raw

    # --- BƯỚC 4: TÍNH TOÁN THỜI GIAN (TIMESTAMPS) ---
    frame_samples = int(processor.sample_rate * processor.frame_size_ms / 1000)
    n_time_frames = int(np.ceil(frame_samples / processor.hop_length))
    step = max(1, int(n_time_frames * 0.75)) 
    time_per_pred = (step * processor.hop_length) / processor.sample_rate
    
    target_names = {0: 'Background', 1: 'Glass Breaking', 2: 'Gunshot', 3: 'Scream'}
    raw_events = []
    
    current_event_id = y_pred_smoothed[0]
    start_time = 0.0

    # Chuyển đổi dãy frame thành các sự kiện thô
    for i in range(1, len(y_pred_smoothed)):
        if y_pred_smoothed[i] != current_event_id:
            end_time = i * time_per_pred
            if current_event_id != 0:
                raw_events.append({
                    'event': target_names[current_event_id],
                    'start_sec': round(start_time, 3),
                    'end_sec': round(end_time, 3),
                    'duration': round(end_time - start_time, 3)
                })
            current_event_id = y_pred_smoothed[i]
            start_time = end_time

    end_time = len(y_pred_smoothed) * time_per_pred
    if current_event_id != 0:
         raw_events.append({
            'event': target_names[current_event_id],
            'start_sec': round(start_time, 3),
            'end_sec': round(end_time, 3),
            'duration': round(end_time - start_time, 3)
        })

    # --- BƯỚC 5: LOGIC HẬU KỲ (GỘP SỰ KIỆN VÀ LỌC RÁC) ---
    merged_events =[]
    for ev in raw_events:
        if not merged_events:
            merged_events.append(ev)
            continue
            
        last_ev = merged_events[-1]
        
        # Nối sự kiện: Nếu cùng loại và cách nhau dưới 0.6 giây -> Gộp làm một
        if ev['event'] == last_ev['event'] and (ev['start_sec'] - last_ev['end_sec']) <= 0.6:
            last_ev['end_sec'] = ev['end_sec']
            last_ev['duration'] = round(last_ev['end_sec'] - last_ev['start_sec'], 3)
        else:
            merged_events.append(ev)

    # Bỏ qua các sự kiện quá ngắn (dưới min_duration) để lọc False Positives
    final_events = [ev for ev in merged_events if ev['duration'] >= min_duration]

    return final_events