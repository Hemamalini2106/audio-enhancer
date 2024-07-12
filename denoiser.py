import numpy as np
import keras
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter

window_length = 255
fft_length = 255
hop_length = 63
frame_length = 8064
debug_flag = False
filtering_flag = True

model = keras.models.load_model('./model.h5')

CHUNK = 8064
RATE = 22050
samp_interv=CHUNK

def convert_to_stft(data):
    data_stft = librosa.stft(data, n_fft=fft_length, hop_length=hop_length)
    data_stft_mag, data_stft_phase = librosa.magphase(data_stft)
    if debug_flag:
        print("STFT shape:")
        print(data_stft_mag.shape)
    data_stft_mag_db = librosa.amplitude_to_db(data_stft_mag, ref=np.max)
    data_stft_mag_db_scaled = (data_stft_mag_db + 80) / 80
    data_stft_mag_db_scaled = np.reshape(data_stft_mag_db_scaled, (1, data_stft_mag_db_scaled.shape[0], data_stft_mag_db_scaled.shape[1], 1))
    return data_stft_mag_db_scaled, data_stft_mag, data_stft_phase

def convert_to_time_domain(predicted_clean, data_stft_phase, data_stft_mag):
    predicted_mag_db_unscaled = (predicted_clean * 80) - 80
    predicted_mag = librosa.db_to_amplitude(predicted_mag_db_unscaled, ref=np.max(data_stft_mag))
    predicted_stft = predicted_mag * data_stft_phase
    predicted_final = librosa.istft(predicted_stft, hop_length=hop_length, length=frame_length)
    if debug_flag:
        print("Predicted final shape: ")
        print(predicted_final.shape)
    return predicted_final

def run_denoiser(noisy_sample):
    data_stft_mag_db_scaled, data_stft_mag, data_stft_phase = convert_to_stft(noisy_sample)
    predicted_clean = model.predict(data_stft_mag_db_scaled)
    predicted_clean = np.reshape(predicted_clean, (predicted_clean.shape[1], predicted_clean.shape[2]))
    output_clean = convert_to_time_domain(predicted_clean, data_stft_phase, data_stft_mag)
    if filtering_flag:
        if np.max(output_clean) < 0.01:
            lo, hi = 300, 1000
            if np.max(output_clean) < 0.005:
                lo, hi = 1000, 1500
            b, a = butter(N=6, Wn=[2 * lo / RATE, 2 * hi / RATE], btype='band')
            x = lfilter(b, a, output_clean)
            output_clean = np.float32(x)
        lo, hi = 50, 2000
        b, a = butter(N=6, Wn=[2 * lo / RATE, 2 * hi / RATE], btype='band')
        x = lfilter(b, a, output_clean)
        output_clean = np.float32(x)
    return output_clean
