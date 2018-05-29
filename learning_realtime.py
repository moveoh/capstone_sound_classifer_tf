'''

 wav파일을 사용한 데이터셋(npz파일)을 활용해서 실제로 학습을 하고 ckpt파일로 모델의변수를 저장하는 부분
 *ckpt에는 weight 와 bias 값만 저장된다! 명심 ㅎㅎ

'''

import numpy as np
import tensorflow as tf
from random import shuffle

# 텐서플로우 모델 생성
n_dim = 193
n_classes = 4
n_hidden_units_one = 300
n_hidden_units_two = 200
n_hidden_units_three = 100

training_epochs = 5000  # 학습 횟수
learning_rate = 0.01  # 학습 비율
sd = 1 / np.sqrt(n_dim)  # standard deviation 표준편차(표본표준편차라 1/root(n))

X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, n_classes])

# 1차 히든 레이어(원소까지 랜덤인 배열을 생성)
W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0, stddev=sd), name="w1")
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0, stddev=sd), name="b1")
# 1차 히든레이어는 Activation 함수로 'Relu' 함수를 쓴다.
h_1 = tf.nn.relu(tf.matmul(X, W_1) + b_1)

# 2차 히든 레이어
W_2 = tf.Variable(tf.random_normal([n_hidden_units_one, n_hidden_units_two], mean=0, stddev=sd), name="w2")
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0, stddev=sd), name="b2")
# 2차 히든레이어는 '하이퍼볼릭탄젠트' 함수를 쓴다.
h_2 = tf.nn.tanh(tf.matmul(h_1, W_2) + b_2)

# 3차 히든 레이어
W_3 = tf.Variable(tf.random_normal([n_hidden_units_two, n_hidden_units_three], mean=0, stddev=sd), name="w3")
b_3 = tf.Variable(tf.random_normal([n_hidden_units_three], mean=0, stddev=sd), name="b3")
# 2차 히든레이어는 'Relu' 함수를 쓴다.
h_3 = tf.nn.relu(tf.matmul(h_2, W_3) + b_3)

# 드롭아웃 과정 추가
keep_prob = tf.placeholder(tf.float32)
h_3_drop = tf.nn.dropout(h_3, keep_prob)

# 최종 evidence 레이어(?? 이거 뭐라고 불러야하지)
W = tf.Variable(tf.random_normal([n_hidden_units_three, n_classes], mean=0, stddev=sd), name="w")
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd), name="b")
y_ = tf.nn.softmax(tf.matmul(h_3_drop, W) + b)


cross_entropy = -tf.reduce_sum(Y*tf.log(tf.clip_by_value(y_, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

#학습이 완료되면 정답률을 체크한다.
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


BATCH_SIZE = 50

# 세션을 켜고, 초기화한다.

data = np.load("Sound_Set/realtime_sound_set.npz")
features = data['X']
labels = data['y']

shuffle_data = True
if shuffle_data:
    c = list(zip(features, labels))
    shuffle(c)
    features, labels = zip(*c)

features = np.array(features)
labels = np.array(labels)


train_features = features[0:int(0.8 * len(features))]
train_labels = labels[0:int(0.8 * len(labels))]

test_features = features[int(0.8 * len(features)):]
test_labels = labels[int(0.8 * len(labels)):]


#print(train_labels)
saver = tf.train.Saver()

with tf.Session() as sess:  # 이러면 끝나고 세션을 자동으로 닫아준다.
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        # 100번마다 정확도가 출력된다.
        train, train_accuracy = sess.run([train_step, accuracy], feed_dict={X: features, Y: labels, keep_prob: 0.5})
        #print(train)
        print("step %d, training accuracy %g" % (epoch, train_accuracy))
        if(train_accuracy > 0.99) :
            print("step %d, training success!" % epoch)
            break

    save_path = saver.save(sess, "saved_realtime/catpstone_checkpoint_model.ckpt")
    writer = tf.summary.FileWriter('mygraph', sess.graph)
    print("The model is saved in file as, : ", save_path)


    TEST_BSIZE = 50
    for i in range(int(len(test_features) / TEST_BSIZE)):
        test_features = test_features[i * TEST_BSIZE:(i + 1) * TEST_BSIZE] 
        test_labels = test_labels[i * TEST_BSIZE:(i + 1) * TEST_BSIZE]

        print(sess.run(accuracy, feed_dict={X: test_features, Y: test_labels, keep_prob: 1}))

















