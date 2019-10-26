import numpy as np
import librosa
from pydub import AudioSegment


def load_wav(file_path, Fs=8000):
    audiofile = AudioSegment.from_file(file_path)
    audiofile = audiofile.set_frame_rate(Fs)
    data = np.fromstring(audiofile._data, np.int16)
    Fs = audiofile.frame_rate

    return data, Fs


def extract_spectrogram(signal, Fs, time_slice=3, time_stride=3):
    """Extract spectrograms of input file"""
    seg = int(time_slice * Fs)
    stride = int(time_stride * Fs)
    melspecgram = librosa.feature.melspectrogram(signal, Fs,
                                                 S=None,
                                                 n_mels=128,
                                                 n_fft=int(0.050*Fs),
                                                 hop_length=int(0.0250*Fs),
                                                 power=2.0)
    spec = np.log10(melspecgram + 1e-08)
    return spec
