import numpy as np
import librosa


# =========================
# 1. PROCESSOR (code của bạn)
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
            return np.array([])

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

        log_mel = librosa.power_to_db(mel, ref=1.0)

        total_time_frames = log_mel.shape[1]

        step = max(1, int(self.n_time_frames * 0.75))

        # padding
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

        frames = []
        start = 0

        while start + self.n_time_frames <= total_time_frames:
            frame = log_mel[:, start:start + self.n_time_frames]
            frames.append(frame)
            start += step

        return np.array(frames, dtype=np.float32)

    def extract_features(self, data_list):
        X, Y = [], []

        for item in data_list:
            frames = self._audio_to_mel_frames(item['audio'])
            if len(frames) == 0:
                continue

            num_frames = frames.shape[0]

            remainder = num_frames % self.seq_length
            if remainder != 0:
                pad_width = self.seq_length - remainder
                frames = np.pad(
                    frames,
                    ((0, pad_width), (0, 0), (0, 0)),
                    mode='constant',
                    constant_values=frames.min()
                )
                num_frames = frames.shape[0]

            num_sequences = num_frames // self.seq_length

            sequences = frames.reshape(
                (num_sequences, self.seq_length, self.n_mels, self.n_time_frames)
            )

            seq_labels = np.full((num_sequences, self.seq_length), item['label'])

            X.append(sequences)
            Y.append(seq_labels)

        if X:
            return np.concatenate(X, axis=0), np.concatenate(Y, axis=0)

        return np.array([]), np.array([])


# =========================
# 2. ĐỌC RAW INMP441
# =========================
def read_inmp441_raw(file_path):
    """
    Đọc file raw từ INMP441 (ESP32 I2S)
    """

    # đọc int32 (phổ biến nhất)
    audio = np.fromfile(file_path, dtype=np.int32)

    # shift để lấy 24-bit thật
    audio = audio >> 14

    # normalize về [-1, 1]
    audio = audio.astype(np.float32) / (2**23)

    return audio


# =========================
# 3. MAIN TEST
# =========================
if __name__ == "__main__":
    file_path = "test.raw"

    audio = read_inmp441_raw(file_path)

    print("Audio stats:")
    print("Min:", np.min(audio))
    print("Max:", np.max(audio))
    print("Shape:", audio.shape)

    data_list = [
        {"audio": audio, "label": 1}  # 1 = gunshot (ví dụ)
    ]

    processor = MIVIASpectrogramProcessor(sample_rate=16000)

    X, Y = processor.extract_features(data_list)

    print("\n=== RESULT ===")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)