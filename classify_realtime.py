import os

import numpy as np
import librosa
import pyaudio
import wave
import tensorflow as tf

'''
마이크를 통해서 소리를 녹음받아서 1초단위로 저장하고 그 파일을 읽어와 각종 특징을 추출 후 (extract_feature()) 
그 특징을 활용하여 classify를 하는데, 이 때 ckpt 파일을 복구하여 학습된 모델에 run 시키고 
정확도와 라벨을 리턴해주는 코드입니다.
'''

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "recording"
term = 0
p = pyaudio.PyAudio()

# 이미 존재하던 파일을 삭제한다
#os.remove('recording.wav')


def get_audio_input_stream():
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    return stream


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz


# 텐서플로우 모델 생성
n_dim = 193
n_classes = 4
n_hidden_units_one = 300
n_hidden_units_two = 200
n_hidden_units_three = 100
sd = 1 / np.sqrt(n_dim)

X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0, stddev=sd), name="w1")
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0, stddev=sd), name="b1")
h_1 = tf.nn.relu(tf.matmul(X, W_1) + b_1)

W_2 = tf.Variable(tf.random_normal([n_hidden_units_one, n_hidden_units_two], mean=0, stddev=sd), name="w2")
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0, stddev=sd), name="b2")
h_2 = tf.nn.tanh(tf.matmul(h_1, W_2) + b_2)

W_3 = tf.Variable(tf.random_normal([n_hidden_units_two, n_hidden_units_three], mean=0, stddev=sd), name="w3")
b_3 = tf.Variable(tf.random_normal([n_hidden_units_three], mean=0, stddev=sd), name="b3")
h_3 = tf.nn.relu(tf.matmul(h_2, W_3) + b_3)

W = tf.Variable(tf.random_normal([n_hidden_units_three, n_classes], mean=0, stddev=sd), name="w")
b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd), name="b")
z = tf.matmul(h_3, W) + b
y_sigmoid = tf.nn.sigmoid(z)
y_ = tf.nn.softmax(z)


init = tf.global_variables_initializer()

# 모델 파라메타 로드
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
saver.restore(sess, "saved_realtime/catpstone_checkpoint_model.ckpt")

frames = []
stream = get_audio_input_stream()

while True:
    try:
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        wf = wave.open(WAVE_OUTPUT_FILENAME + ".wav", 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

        mfccs, chroma, mel, contrast, tonnetz = extract_feature(WAVE_OUTPUT_FILENAME + ".wav")
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])

        y_hat, sigmoid = sess.run([y_, y_sigmoid], feed_dict={X: ext_features.reshape(1, -1)})
        index = np.argmax(y_hat)

        # 정확도 print
        print(sigmoid)

        # 소리 인덱스 리턴
        if index == 0:
            print("조용한 상황")
        elif index == 1:
            print("혼자 말하는 상황")
        elif index == 2:
            print("시끄러운 상황")
        elif index == 3:
            print("길거리(차도) 상황")

        term += 1
        frames.clear()

    except IOError as e:
        print(e)
        stream = get_audio_input_stream()
        continue

stream.stop_stream()
stream.close()
p.terminate()
wf.close()
