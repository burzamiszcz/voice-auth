import os
import pickle
import time
import wave

import librosa
import numpy as np
import pyaudio
import python_speech_features as mfcc
import soundfile as sf
from PyQt5.QtCore import pyqtSignal, QObject
from scipy.io.wavfile import read
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture


class VoiceVerificator(QObject):
    finished = pyqtSignal()
    is_recording = pyqtSignal(bool)
    result = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 512
        self.RECORD_SECONDS = 3
        self.audio = pyaudio.PyAudio()
        self.username = 'default'
        self.input_device_index = 0
        self.verification_filename = None
        self.threshold = -35

    def calculate_delta(self, arr):
        rows, cols = arr.shape
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
            deltas[i] = (arr[index[0][0]] - arr[index[0][1]] + (2 * (arr[index[1][0]] - arr[index[1][1]]))) / 10
        return deltas

    def extract_features(self, audio, rate):
        mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, nfft=1200, appendEnergy=True)
        mfcc_feature = preprocessing.scale(mfcc_feature)
        # print(mfcc_feature)
        delta = self.calculate_delta(mfcc_feature)
        combined = np.hstack((mfcc_feature, delta))
        return combined

    def get_input_device_list(self):
        devices = []
        audio = pyaudio.PyAudio()
        for i in range(0, audio.get_host_api_info_by_index(0).get('deviceCount')):
            if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                devices.append(audio.get_device_info_by_host_api_device_index(0, i).get('name'))
        return devices

    def record_audio_train(self):
        for count in range(5):
            self.is_recording.emit(True)
            print('started')
            audio = pyaudio.PyAudio()
            stream = audio.open(format=self.FORMAT,
                                channels=self.CHANNELS,
                                rate=self.RATE,
                                input=True,
                                input_device_index=self.input_device_index,
                                frames_per_buffer=self.CHUNK)
            Recordframes = []
            for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                data = stream.read(self.CHUNK)
                Recordframes.append(data)
            stream.stop_stream()
            stream.close()
            audio.terminate()
            OUTPUT_FILENAME = self.username + '-sample' + str(count) + '.wav'
            WAVE_OUTPUT_FILENAME = os.path.join('training_set', OUTPUT_FILENAME)
            trainedfilelist = open('training_set_addition.txt', 'a')
            trainedfilelist.write(OUTPUT_FILENAME + "\n")
            waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            waveFile.setnchannels(self.CHANNELS)
            waveFile.setsampwidth(audio.get_sample_size(self.FORMAT))
            waveFile.setframerate(self.RATE)
            waveFile.writeframes(b''.join(Recordframes))
            waveFile.close()

            audio_file = WAVE_OUTPUT_FILENAME
            audio, sr = librosa.load(audio_file, sr=8000, mono=True)
            clip = librosa.effects.trim(audio, top_db=5)
            sf.write(WAVE_OUTPUT_FILENAME, clip[0], sr)
            print('ended')
            self.is_recording.emit(False)
            time.sleep(1.0)
        self.train_model()
        self.finished.emit()

    def train_model(self):
        source = 'training_set\\'
        dest = 'trained_models\\'
        train_file = 'training_set_addition.txt'
        path = 'training_set'
        print(os.listdir('training_set'))
        user_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and self.username + '-' in f]
        print(user_files)
        file_paths = open(train_file, 'r')
        count = 1
        features = np.asarray(())
        for path in user_files:
            path = path.strip()
            print(path)
            sr, audio = read(source + path)
            print(sr)
            vector = self.extract_features(audio, sr)

            if features.size == 0:
                features = vector
            else:
                features = np.vstack((features, vector))

            if count == 5:
                gmm = GaussianMixture(n_components=6, max_iter=200, covariance_type='diag', n_init=3)
                gmm.fit(features)

                # dumping the trained gaussian model
                picklefile = path.split("-")[0] + '.gmm'
                pickle.dump(gmm, open(dest + picklefile, 'wb'))
                print('+ modeling completed for speaker:', picklefile, ' with data point = ', features.shape)
                features = np.asarray(())
                count = 0
            count = count + 1

    # nagrywa jedno audio do testowania
    def record_audio_test(self):
        self.is_recording.emit(True)
        audio = pyaudio.PyAudio()
        stream = audio.open(format=self.FORMAT, channels=self.CHANNELS,
                            rate=self.RATE, input=True, input_device_index=self.input_device_index,
                            frames_per_buffer=self.CHUNK)
        print("recording started")
        Recordframes = []
        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            data = stream.read(self.CHUNK)
            Recordframes.append(data)
        print("recording stopped")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        OUTPUT_FILENAME = self.username + '-sample.wav'
        WAVE_OUTPUT_FILENAME = os.path.join('testing_set', OUTPUT_FILENAME)
        trainedfilelist = open('testing_set_addition.txt', 'w')
        trainedfilelist.write(OUTPUT_FILENAME + "\n")
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(self.CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(self.FORMAT))
        waveFile.setframerate(self.RATE)
        waveFile.writeframes(b''.join(Recordframes))
        waveFile.close()

        audio_file = WAVE_OUTPUT_FILENAME
        audio, sr = librosa.load(audio_file, sr=8000, mono=True)
        clip = librosa.effects.trim(audio, top_db=5)
        sf.write(WAVE_OUTPUT_FILENAME, clip[0], sr)
        self.is_recording.emit(False)
        time.sleep(1)
        self.test_model()
        self.finished.emit()

    # zwraca wartość .score() dla użytkownika
    def test_model2(self):
        source = 'testing_set\\'
        modelpath = 'trained_models\\'
        test_file = 'testing_set_addition.txt'
        file_paths = open(test_file, 'r')

        gmm_files = [os.path.join(modelpath,fname) for fname in
                    os.listdir(modelpath) if fname.endswith('.gmm')]
        
        #Load the Gaussian Mixture Models
        models = [pickle.load(open(fname,'rb')) for fname in gmm_files]
        speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname
                    in gmm_files]
        
        # Read the test directory and get the list of test audio files

        if self.verification_filename is None:
            path = self.username + '-sample.wav'
        else:
            path = self.verification_filename
        path = path.strip()
        print(path)
        sr, audio = read(source + path)
        vector = self.extract_features(audio,sr)
        
        log_likelihood = np.zeros(len(models)) 
        
        for i in range(len(models)):
            gmm = models[i]  #checking with each model one by one
            scores = np.array(gmm.score(vector))
            print('\nscores\n')
            print(scores.sum())
            # print(math.exp(scores))
            # print('\npredict_proba\n')
            # print(np.round(gmm.predict_proba(vector), 2))
            log_likelihood[i] = scores.sum()
        
        winner = np.argmax(log_likelihood)
        if speakers[winner] == self.username:
            self.result.emit('Zweryfikowano pomyślnie')
        else:
            self.result.emit('Weryfikacja nie powiodła się')

        print("\tdetected as - ", speakers[winner])
        time.sleep(1.0)

    def test_model(self):
        source = 'testing_set\\'
        modelpath = 'trained_models\\'

        gmm_file_path = os.path.join(modelpath, self.username + '.gmm')
        gmm = pickle.load(open(gmm_file_path, 'rb'))

        if self.verification_filename is None:
            verification_file_path = source + self.username + '-sample.wav'
        else:
            verification_file_path = source + self.verification_filename
        sr, audio = read(verification_file_path)

        vector = self.extract_features(audio, sr)
        log_likelihood = np.array(gmm.score(vector)).sum()

        print(verification_file_path)
        print('\nExperiment for ' + self.username)
        print('Model = ' + self.username + '.gmm')
        if self.verification_filename is None:
            print('Verification File = ' + self.username + '-sample.wav')
        else:
            print('Verification File = ' + self.verification_filename)
        print('log likelihood = ' + str(log_likelihood))
        print('threshold = ' + str(self.threshold))

        if log_likelihood > self.threshold:
            print('Zweryfikowano pomyślnie\n')
            self.result.emit('Zweryfikowano pomyślnie')
        else:
            print('Weryfikacja nie powiodła się\n')
            self.result.emit('Weryfikacja nie powiodła się')
        time.sleep(1.0)

    def test_file_recording(self):
        self.test_model()
        self.verification_filename = None
        self.finished.emit()

    # zwraca wartość .score() dla użytkownika
    def test_model_for_experiment(self, verification_username):
        source = 'testing_set\\'
        modelpath = 'trained_models\\'

        gmm_file_path = os.path.join(modelpath, self.username + '.gmm')
        gmm = pickle.load(open(gmm_file_path, 'rb'))

        verification_file_path = source + verification_username + '-sample.wav'
        sr, audio = read(verification_file_path)

        vector = self.extract_features(audio, sr)
        self.log_likelihood = np.array(gmm.score(vector)).sum()

        print('\nExperiment for ' + self.username)
        print('Model = ' + self.username + '.gmm')
        print('Verification File = ' + verification_username + '-sample.wav')
        print('log likelihood = ' + str(self.log_likelihood))
        print('threshold = ' + str(self.threshold))

        if self.log_likelihood > self.threshold:
            print('Zweryfikowano pomyślnie\n')
        else:
            print('Weryfikacja nie powiodła się\n')
