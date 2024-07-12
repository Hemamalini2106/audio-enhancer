import numpy as np
import librosa
import denoiser as df
import soundfile as sf

noisy_file = './noise_audio.wav'
clean_file_name = './output/Denoised.wav'
CHUNK = 8064
RATE = 22050
samp_interv = CHUNK

if __name__ == "__main__":
    print("\n\n\n")
    noisy_sample_test_split = []
    clean_audio_array = []

    noisy_sample_test, noise_sample_sr = librosa.load(noisy_file, sr=RATE)
    for j in range(samp_interv, len(noisy_sample_test), samp_interv):
        k = j - samp_interv
        noisy_sample_test_split.append(noisy_sample_test[k:j])

    noisy_sample_test_split = np.array(noisy_sample_test_split)
    print(noisy_sample_test_split.shape)

    for i in range(len(noisy_sample_test_split)):
        clean_audio = df.run_denoiser(noisy_sample_test_split[i])
        clean_audio_array.append(clean_audio)

    clean_audio_array = np.array(clean_audio_array)
    clean_wav = np.reshape(clean_audio_array, (clean_audio_array.shape[0] * clean_audio_array.shape[1]))
    print("Output = " + str(clean_wav.shape))
    sf.write(clean_file_name, clean_wav, RATE)
