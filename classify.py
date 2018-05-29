'''

실제로 소리를 녹음하여 test폴더에 넣었고(학습한 데이터와 전혀 다른데이터)
ckpt 파일을 복구하여 학습된 모델에 run 시키고 정확도와 라벨을 리턴해주는 코드입니다.

Labels

0 - 조용한 상황
1 - 혼자 이야기하는 상황
2 - 시끄러운 상황
3 - 영화관
4 - 길거리(차도)

'''

import numpy as np
import tensorflow as tf
import librosa
import winsound


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
n_classes = 5
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
saver.restore(sess, "saved/catpstone_checkpoint_model.ckpt")



for i in range(4) :
    file_path = "Sound_Set_test/test-" + str(i) + ".wav"

    winsound.PlaySound(file_path, winsound.SND_FILENAME)
    audio_file = file_path
    mfccs, chroma, mel, contrast, tonnetz = extract_feature(audio_file)
    x_data = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    y_hat, sigmoid = sess.run([y_, y_sigmoid], feed_dict={X: x_data.reshape(1, -1)})
    index = np.argmax(y_hat)

    # 각 라벨이 일치하는 정도
    print(y_hat)
    #print(sigmoid)
    # 소리 인덱스 리턴
    if index == 0:
        print("조용한 상황")
    elif index == 1:
        print("혼자 말하는 상황")
    elif index == 2:
        print("시끄러운 상황")
    elif index == 3:
        print("버스킹")
    elif index == 4:
        print("길거리(차도) 상황")