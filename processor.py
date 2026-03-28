import librosa
import numpy as np


class MIVIASpectrogramProcessor:
    """
    Processes audio slices into mel‑spectrogram sequences.
    - Tính STFT cho toàn bộ audio slice, lấy log‑mel.
    - Chia thành các frame với 25% overlap.
    - Ghép các frame thành sequences (seq_length).
    """
    def __init__(self, sample_rate=16000, frame_size_ms=50, seq_length=10,
                 n_mels=128, n_fft=1024, hop_length=256):
        self.sample_rate = sample_rate
        self.frame_size_ms = frame_size_ms
        self.seq_length = seq_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Số mẫu trong một frame
        self.frame_samples = int(self.sample_rate * self.frame_size_ms / 1000)
        # Số khung thời gian của spectrogram trong một frame
        self.n_time_frames = int(np.ceil(self.frame_samples / self.hop_length))

    def _audio_to_mel_frames(self, audio):
        """
        Tính log-mel spectrogram và cắt thành các frame với 25% overlap.
        Nếu bị lẻ, sẽ padding thêm khoảng lặng để không mất dữ liệu ở cuối.
        """
        if len(audio) == 0:
            return np.array([])
    
        # STFT → Mel → Log
        D = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window='hann',
            center=False
        )
    
        mel = librosa.feature.melspectrogram(
            S=np.abs(D) ** 2,
            sr=self.sample_rate,
            n_mels=self.n_mels
        )
        
        # FIX: Dùng ref=1.0 thay vì np.max để giữ nguyên cường độ tuyệt đối. 
        # Nếu dùng np.max, khoảng lặng sẽ bị khuếch đại lên bằng âm thanh tiếng súng!
        log_mel = librosa.power_to_db(mel, ref=1.0)
        
        total_time_frames = log_mel.shape[1]
        
        # --- Tính Step cho 25% Overlap (Shift 75%) ---
        step = max(1, int(self.n_time_frames * 0.75)) 
    
        # --- Padding toán học chuẩn cho Sliding Window ---
        # Đảm bảo cửa sổ cuối cùng vừa khít không bị thiếu frame nào
        if total_time_frames < self.n_time_frames:
            pad_width = self.n_time_frames - total_time_frames
        else:
            remainder = (total_time_frames - self.n_time_frames) % step
            pad_width = (step - remainder) if remainder != 0 else 0
            
        if pad_width > 0:
            log_mel = np.pad(
                log_mel,
                ((0, 0), (0, pad_width)),
                mode='constant',
                constant_values=log_mel.min()
            )
            
        total_time_frames = log_mel.shape[1]
    
        # --- Sliding window ---
        frames =[]
        start = 0
    
        while start + self.n_time_frames <= total_time_frames:
            frame = log_mel[:, start:start + self.n_time_frames]
            frames.append(frame)
            start += step
    
        return np.array(frames, dtype=np.float32)

    def extract_features(self, data_list, verbose=True):
        """Xử lý danh sách các slice audio thành X (sequences) và Y (labels many-to-many)."""
        if verbose:
            print(f"Processing {len(data_list)} audio slices into spectrogram sequences...")
        X, Y = [],[]
        
        for item in data_list:
            frames = self._audio_to_mel_frames(item['audio'])
            if len(frames) == 0:
                continue
                
            num_frames = frames.shape[0]
            
            # --- Pad frames so it divides perfectly by seq_length ---
            remainder = num_frames % self.seq_length
            if remainder != 0:
                pad_width = self.seq_length - remainder
                # Pad frames with the minimum value representing silence
                frames = np.pad(
                    frames, 
                    ((0, pad_width), (0, 0), (0, 0)), 
                    mode='constant', 
                    constant_values=frames.min()
                )
                num_frames = frames.shape[0]
                
            num_sequences = num_frames // self.seq_length
            sequences = frames.reshape((num_sequences, self.seq_length, self.n_mels, self.n_time_frames))
            
            # Vì audio đã được cắt riêng biệt từng Event/Background trong Loader, 
            # gán nhãn hàng loạt cho cả chuỗi là hoàn toàn chính xác.
            seq_labels = np.full((num_sequences, self.seq_length), item['label'])
            
            X.append(sequences)
            Y.append(seq_labels)

        if X:
            X_concat = np.concatenate(X, axis=0)
            Y_concat = np.concatenate(Y, axis=0)
            if verbose:
                print(f"Finished. X shape: {X_concat.shape}, Y shape: {Y_concat.shape}")
            return X_concat, Y_concat
        return np.array([]), np.array([])