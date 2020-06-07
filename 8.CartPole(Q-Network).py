import numpy as np
import tensorflow as tf
import gym

env = gym.make('CartPole-v0')
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

learning_rate = 1e-1
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

X = tf.compat.v1.placeholder(tf.float32, [None, input_size], name="input_x")
W1 = tf.compat.v1.get_variable("W1", shape=[input_size, output_size],
                               initializer=tf.keras.initializers.glorot_normal)
Qpred = tf.compat.v1.matmul(X, W1)

Y = tf.compat.v1.placeholder(shape=[None, output_size], dtype=tf.float32)

loss = tf.compat.v1.reduce_sum(tf.compat.v1.square(Y - Qpred))
train = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

num_episodes = 2000
dis = 0.9
rList = []

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(num_episodes):
        rAll = 0
        step_count = 0
        s = env.reset()
        done = False

        while not done:
            step_count += 1
            x = np.reshape(s, [1, input_size])
            Qs = sess.run(Qpred, feed_dict={X: x})
            if np.random.rand(1) < e:
                a = env.action_space.sample()
            else:
                a = np.argmax(Qs)
            s1, reward, done, info = env.step(a)

            if done:
                Qs[0, a] = -100
            else:
                Qs1 = sess.run(Qpred, feed_dict={X: x})
                Qs[0, a] = reward + dis * np.max(Qs1)

            sess.run(train, feed_dict={X: x, Y: Qs})
            s = s1
        rList.append(step_count)
        print("Episodes: {} steps: {}".format(i, step_count))

        if len(rList) > 10 and np.mean(rList[-10:]) > 500:
            break
    observation = env.reset()
    reward_sum = 0

    while True:
        env.render()

        x = np.reshape(observation, [1, input_size])
        Qs = sess.run(Qpred, feed_dict={X: x})
        a = np.argmax(Qs)

        observation, reward, done, _ = env.step(a)
        reward_sum += reward
        if done:
            print("Total score: {}".format(reward_sum))
            break
