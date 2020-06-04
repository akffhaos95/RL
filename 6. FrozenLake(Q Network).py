import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def one_hot(x):
    return np.identity(16)[x : x+1]

env = gym.make('FrozenLake-v0')
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1

X = tf.compat.v1.placeholder(tf.float32, shape=([1, input_size]))
W = tf.compat.v1.Variable(tf.compat.v1.random_uniform([input_size, output_size], 0, 0.01))

Qpred = tf.compat.v1.matmul(X, W)
Y = tf.compat.v1.placeholder(shape=[1, output_size], dtype=tf.float32)

loss = tf.compat.v1.reduce_sum(tf.square(Y - Qpred))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)

dis = .99
num_episodes = 2000
rList = []

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(num_episodes):
        s = env.reset()
        e = 1. / ((i/50) + 10)
        rAll = 0
        done = False
        local_loss = []

        while not done:
            Qs = sess.run(Qpred, feed_dict = {X: one_hot(s)})
            if np.random.rand(1) < e:
                a = env.action_space.sample()
            else:
                a = np.argmax(Qs)
            s1, reward, done, _ = env.step(a)
            if done:
                Qs[0, a] = reward
            else :
                Qs1 = sess.run(Qpred, feed_dict= {X: one_hot(s1)})
                Qs[0, a] = reward + dis * np.max(Qs1)
            sess.run(train, feed_dict = {X: one_hot(s), Y: Qs})
            rAll += reward
            s = s1
        rList.append(rAll)

print("Percent of successful episodes: " + str(sum(rList)/num_episodes) + "%")
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
