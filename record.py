import os
import pickle
import time
import wave

import librosa
import numpy as np
import pyaudio
import python_speech_features as mfcc
import soundfile as sf
from scipy.io.wavfile import read
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture

clear = lambda: os.system('cls')
clear()

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 512
RECORD_SECONDS = 3
audio = pyaudio.PyAudio()


def calculate_delta(array):
    rows, cols = array.shape
    # print(rows)
    # print(cols)
    deltas = np.zeros((rows, 20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i - j < 0:
                first = 0
            else:
                first = i - j
            if i + j > rows - 1:
                second = rows - 1
            else:
                second = i + j
            index.append((second, first))
            j += 1
        deltas[i] = (array[index[0][0]] - array[index[0][1]] + (2 * (array[index[1][0]] - array[index[1][1]]))) / 10
    return deltas


def extract_features(audio, rate):
    mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, nfft=1200, appendEnergy=True)
    mfcc_feature = preprocessing.scale(mfcc_feature)
    # print(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature, delta))
    return combined


def record_audio_train():
    Name = (input("Podaj nazwę użytkownika:"))
    for count in range(5):
        device_index = 2
        audio = pyaudio.PyAudio()
        if count == 0:
            print("----------------------Wybierz urządzenie do nagrywania---------------------")
            info = audio.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            for i in range(0, numdevices):
                if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                    print("ID ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
            print("-------------------------------------------------------------")
            index = int(input())
        print("Nagrywanie przez  " + str(index))
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True, input_device_index=index,
                            frames_per_buffer=CHUNK)
        print("Nagrywanie rozpoczęte - czas 3 sekundy")
        Recordframes = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            Recordframes.append(data)
        print("Nagrywanie zakończone")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        OUTPUT_FILENAME = Name + "-sample" + str(count) + ".wav"
        WAVE_OUTPUT_FILENAME = os.path.join("training_set", OUTPUT_FILENAME)
        trainedfilelist = open("training_set_addition.txt", 'a')
        trainedfilelist.write(OUTPUT_FILENAME + "\n")
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(Recordframes))
        waveFile.close()

        audio_file = WAVE_OUTPUT_FILENAME
        audio, sr = librosa.load(audio_file, sr=8000, mono=True)
        clip = librosa.effects.trim(audio, top_db=5)
        sf.write(WAVE_OUTPUT_FILENAME, clip[0], sr)


def train_model():
    source = "training_set\\"
    dest = "trained_models\\"
    train_file = "training_set_addition.txt"
    file_paths = open(train_file, 'r')
    count = 1
    features = np.asarray(())
    for path in file_paths:
        path = path.strip()
        print(path)
        sr, audio = read(source + path)
        print(sr)
        vector = extract_features(audio, sr)

        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

        if count == 5:
            gmm = GaussianMixture(n_components=6, max_iter=200, covariance_type='diag', n_init=3)
            gmm.fit(features)

            # dumping the trained gaussian model
            picklefile = path.split("-")[0] + ".gmm"
            pickle.dump(gmm, open(dest + picklefile, 'wb'))
            print('+ modeling completed for speaker:', picklefile, " with data point = ", features.shape)
            features = np.asarray(())
            count = 0
        count = count + 1


# nagrywa jedno audio do testowania
def record_audio_test():
    device_index = 2
    audio = pyaudio.PyAudio()
    print("----------------------record device list---------------------")
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
    print("-------------------------------------------------------------")
    index = int(input())
    print("recording via index " + str(index))
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, input_device_index=index,
                        frames_per_buffer=CHUNK)
    print("recording started")
    Recordframes = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        Recordframes.append(data)
    print("recording stopped")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    OUTPUT_FILENAME = "sample.wav"
    WAVE_OUTPUT_FILENAME = os.path.join("testing_set", OUTPUT_FILENAME)
    trainedfilelist = open("testing_set_addition.txt", 'a')
    trainedfilelist.write(OUTPUT_FILENAME + "\n")
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(Recordframes))
    waveFile.close()

    audio_file = WAVE_OUTPUT_FILENAME
    audio, sr = librosa.load(audio_file, sr=8000, mono=True)
    clip = librosa.effects.trim(audio, top_db=5)
    sf.write(WAVE_OUTPUT_FILENAME, clip[0], sr)


# zwraca wartość .score() dla użytkownika
def test_model(username="lol"):
    source = "testing_set\\"
    modelpath = "trained_models\\"
    test_file = "testing_set_addition.txt"
    file_paths = open(test_file, 'r')

    gmm_files = [os.path.join(modelpath, fname) for fname in
                 os.listdir(modelpath) if fname.endswith('.gmm')]

    gmm_file = os.path.join(modelpath, f"{username}.gmm")

    # Load the Gaussian gender Models
    # models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
    model = pickle.load(open(gmm_file, 'rb'))
    speaker = gmm_file.split("\\")[-1].split(".gmm")[0]

    path = 'sample.wav'
    print(path)
    sr, audio = read(source + path)
    vector = extract_features(audio, sr)

    log_likelihood = np.zeros(1)

    gmm = model  # checking with each model one by one
    scores = np.array(gmm.score(vector))
    log_likelihood = scores.sum()

    print(log_likelihood)
    print("\twynik dla - ", speaker)
    time.sleep(1.0)


# record_audio_train()

# train_model()
# x = int(input('podaj haslo'))
#
# if x == 1:
#     record_audio_test()
# else:
#     test_model()

def play_recorded_files_for(username):
    path = 'training_set'
    print(os.listdir('training_set'))
    user_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and username in f]
    print(user_files)

    for file in user_files:
        wf = wave.open(os.path.join(path, file), 'rb')

        p = pyaudio.PyAudio()

        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        data = wf.readframes(CHUNK)

        while data:
            stream.write(data)
            data = wf.readframes(CHUNK)

play_recorded_files_for('dariusz_owoce')
