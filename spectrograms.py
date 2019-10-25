import numpy as np
import librosa
from pydub import AudioSegment


def load_wav(file_path, Fs=8000):
    audiofile = AudioSegment.from_file(file_path)
    audiofile = audiofile.set_frame_rate(Fs)
    if audiofile.sample_width == 2:
        data = np.fromstring(audiofile._data, np.int16)
    elif audiofile.sample_width == 4:
        data = np.fromstring(audiofile._data, np.int32)
    Fs = audiofile.frame_rate
    x = []
    for chn in range(audiofile.channels):
        x.append(data[chn::audiofile.channels])
    x = np.array(x).T
    if x.ndim == 2:
        if x.shape[1] == 1:
            x = x.flatten()

    return x, Fs


def extract_spectrogram(signal, Fs, time_slice=3, time_stride=3):
    """Extract spectrograms of input file"""
    signal = signal.astype('float')
    seg = int(time_slice * Fs)
    stride = int(time_stride * Fs)
    eps = np.finfo(float).eps
    signal = (signal - signal.mean()) / (signal.std() + eps)
    melspecgram = librosa.feature.melspectrogram(signal, Fs,
                                                 S=None,
                                                 n_mels=128,
                                                 n_fft=int(0.050*Fs),
                                                 hop_length=int(0.0250*Fs),
                                                 power=2.0)
    spec = np.log10(melspecgram + eps)
    return spec
