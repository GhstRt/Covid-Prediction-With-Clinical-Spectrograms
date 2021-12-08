import matplotlib.pyplot as plot
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tensorflow.keras.models import load_model
from scipy.io import wavfile
import numpy as np
import cv2
import sounddevice as sd
import librosa.display
import librosa

def processing(file):
    y, sr = librosa.load(file)
    window_size = 1024
    window = np.hanning(window_size)
    stft = librosa.core.spectrum.stft(y, n_fft=window_size, hop_length=512, window=window)
    out = 2 * np.abs(stft) / np.sum(window)
    fig = plot.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax, cmap="nipy_spectral")
    fig.savefig(file+".jpg")
    return file+".jpg"

def prediction(dosya):
    new_model = load_model('best_mod.h5')
    image = cv2.imread(dosya)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.array(image)
    image = image.astype('float32')
    image /= 255
    image = image.reshape(1, 224, 224, 3)
    result = new_model.predict(image)
    return result

def record(filename):
    fs = 44100
    seconds = 3
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()
    wavfile.write(filename+".wav", fs, myrecording)
    return processing(filename+".wav")