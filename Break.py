import tensorflow as tf
import gym
import numpy as np
import random as ran
import datetime
import matplotlib.pyplot as plt
from collections import deque
from skimage.transform import resize
from skimage.color import rgb2gray

plt.ion()
env = gym.make('BreakoutDeterministic-v4')

MINIBATCH_SIZE = 32
HISTORY_SIZE = 4
TRAIN_START = 50000
FINAL_EXPLORATION = 0.1
TARGET_UPDATE = 10000
MEMORY_SIZE = 400000
EXPLORATION = 1000000
START_EXPLORATION = 1.
INPUT = env.observation_space.shape
OUTPUT = env.action_space.n
HEIGHT = 84
WIDTH = 84
LEARNING_RATE = 0.00025
DISCOUNT = 0.99
EPSILON = 0.01
MOMENTUM = 0.95

model_path = "save/Breakout.ckpt"

def cliped_error(error):
    return tf.where(tf.abs(error) < 1.0, 0.5 * tf.square(error), tf.abs(error) - 0.5)

def pre_proc(X):
    x = np.uint8(resize(rgb2gray(X), (HEIGHT, WIDTH), mode='reflect') * 255)
    return x

def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
    op_holder = []
    src_vars = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)
    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))
    return op_holder

def get_init_state(history, s):
    for i in range(HISTORY_SIZE):
        history[:, :, i] = pre_proc(s)

def get_game_type(count, l, no_life_game, start_live):
    if count == 1:
        start_live = l['ale.lives']
        if start_live == 0:
            no_life_game = True
        else:
            no_life_game = False
    return [no_life_game, start_live]

def get_terminal(start_live, l, reward, no_life_game, ter):
    if no_life_game:
        if reward < 0:
            ter = True
    else:
        if start_live > l['ale.lives']:
            ter = True
            start_live = l['ale.lives']
    return [ter, start_live]

def train_minibatch(mainDQN, targetDQN, minibatch):
    s_stack = []
    a_stack = []
    r_stack = []
    s1_stack = []
    d_stack = []

    for s_r, a_r, r_r, d_r in minibatch:
        s_stack.append(s_r[:, :, :4])
        a_stack.append(a_r)
        r_stack.append(r_r)
        s1_stack.append(s_r[:, :, 1:])
        d_stack.append(d_r)
    d_stack = np.array(d_stack) + 0
    Q1 = targetDQN.get_q(np.array(s1_stack))
    y = r_stack + (1 - d_stack) * DISCOUNT * np.max(Q1, axis=1)
    mainDQN.sess.run(mainDQN.train, feed_dict={mainDQN.X: np.float32(np.array(s_stack) / 255.), mainDQN.Y: y, mainDQN.a: a_stack})

def plot_data(epoch, epoch_score, average_reward, epoch_Q, average_Q, mainDQN):
    plt.clf()
    epoch_score.append(np.mean(average_reward))
    epoch_Q.append(np.mean(average_Q))

    plt.subplot(211)
    plt.axis([0, epoch, 0, np.max(epoch_Q) * 6 / 5])
    plt.xlabel('Training Epochs')
    plt.ylabel('Average Action Value(Q)')
    plt.plot(epoch_Q)

    plt.subplot(212)
    plt.axis([0, epoch, 0, np.max(epoch_score) * 6 / 5])
    plt.xlabel('Training Epochs')
    plt.ylabel('Average Reward per Episode')
    plt.plot(epoch_score, "r")

    plt.pause(0.05)
    plt.savefig("graph/{} epoch".format(epoch - 1))

    save_path = mainDQN.saver.save(mainDQN.sess, model_path, global_step=(epoch - 1))
    print("Model(epoch :", epoch, ") saved in file: ", save_path, " Now time : ", datetime.datetime.now())

class DQNAgent:
    def __init__(self, sess, HEIGHT, WIDTH, HISTORY_SIZE, OUTPUT, NAME='main'):
        self.sess = sess
        self.height = HEIGHT
        self.width = WIDTH
        self.history_size = HISTORY_SIZE
        self.output = OUTPUT
        self.name = NAME
        self.build_network()

    def build_network(self):
        with tf.compat.v1.variable_scope(self.name):
            self.X = tf.compat.v1.placeholder('float', [None, self.height, self.width, self.history_size])
            self.Y = tf.compat.v1.placeholder('float', [None])
            self.a = tf.compat.v1.placeholder('int64', [None])

            f1 = tf.compat.v1.get_variable("f1", shape=[8, 8, 4, 32], initializer=tf.keras.initializers.glorot_normal(seed=None))
            f2 = tf.compat.v1.get_variable("f2", shape=[4, 4, 32, 64], initializer=tf.keras.initializers.glorot_normal(seed=None))
            f3 = tf.compat.v1.get_variable("f3", shape=[3, 3, 64, 64], initializer=tf.keras.initializers.glorot_normal(seed=None))
            w1 = tf.compat.v1.get_variable("w1", shape=[7 * 7 * 64, 512], initializer=tf.keras.initializers.glorot_normal(seed=None))
            w2 = tf.compat.v1.get_variable("w2", shape=[512, OUTPUT], initializer=tf.keras.initializers.glorot_normal(seed=None))

            c1 = tf.compat.v1.nn.relu(tf.nn.conv2d(self.X, f1, strides=[1, 4, 4, 1], padding="VALID"))
            c2 = tf.compat.v1.nn.relu(tf.nn.conv2d(c1, f2, strides=[1, 2, 2, 1], padding="VALID"))
            c3 = tf.compat.v1.nn.relu(tf.nn.conv2d(c2, f3, strides=[1, 1, 1, 1], padding='VALID'))

            l1 = tf.compat.v1.reshape(c3, [-1, w1.get_shape().as_list()[0]])
            l2 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(l1, w1))
            self.Q_pre = tf.compat.v1.matmul(l2, w2)
        a_one_hot = tf.one_hot(self.a, self.output, 1.0, 0.0)
        q_val = tf.compat.v1.reduce_sum(tf.multiply(self.Q_pre, a_one_hot), reduction_indices=1)
        error = cliped_error(self.Y - q_val)
        self.loss = tf.compat.v1.reduce_mean(error)
        optimizer = tf.compat.v1.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=EPSILON)
        self.train = optimizer.minimize(self.loss)
        self.saver = tf.compat.v1.train.Saver(max_to_keep=None)
    def get_q(self, history):
        return self.sess.run(self.Q_pre, feed_dict={self.X: np.reshape(np.float32(history / 255.), [-1, 84, 84, 4])})

    def get_action(self, q, e):
        if e > np.random.rand(1):
            action = np.random.randint(self.output)
        else:
            action = np.argmax(q)
        return action

def main():
    with tf.compat.v1.Session() as sess:
        mainDQN = DQNAgent(sess, HEIGHT, WIDTH, HISTORY_SIZE, OUTPUT, NAME='main')
        targetDQN = DQNAgent(sess, HEIGHT, WIDTH, HISTORY_SIZE, OUTPUT, NAME='target')
        sess.run(tf.compat.v1.global_variables_initializer())
        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)
        recent_rlist = deque(maxlen=100)
        e = 1.
        episode, epoch, frame = 0, 0, 0
        epoch_score, epoch_Q = deque(), deque()
        average_Q, average_reward = deque(), deque()
        epoch_on = False
        no_life_game = False
        replay_memory = deque(maxlen=MEMORY_SIZE)
        while epoch <= 200:
            episode += 1
            history = np.zeros([84, 84, 5], dtype=np.uint8)
            rall, count = 0, 0
            d = False
            ter = False
            start_lives = 0
            s = env.reset()
            get_init_state(history, s)
            while not d:
                frame += 1
                count += 1
                if e > FINAL_EXPLORATION and frame > TRAIN_START:
                    e -= (START_EXPLORATION - FINAL_EXPLORATION) / EXPLORATION
                Q = mainDQN.get_q(history[:, :, :4])
                average_Q.append(np.max(Q))
                action = mainDQN.get_action(Q, e)
                s1, r, d, l = env.step(action)
                ter = d
                reward = np.clip(r, -1, 1)
                no_life_game, start_lives = get_game_type(count, l, no_life_game, start_lives)
                ter, start_lives = get_terminal(start_lives, l, reward, no_life_game, ter)
                history[:, :, 4] = pre_proc(s1)
                replay_memory.append((np.copy(history[:, :, :]), action, reward, ter))
                history[:, :, :4] = history[:, :, 1:]
                rall += r
                if frame > TRAIN_START:
                    minibatch = ran.sample(replay_memory, MINIBATCH_SIZE)
                    train_minibatch(mainDQN, targetDQN, minibatch)
                    if frame % TARGET_UPDATE == 0:
                        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
                        sess.run(copy_ops)
                if (frame - TRAIN_START) % 50000 == 0:
                    epoch_on = True
            recent_rlist.append(rall)
            average_reward.append(rall)
            print("Episode:{0:6d} | Frames:{1:9d} | Steps:{2:5d} | Reward:{3:3.0f} | e-greedy:{4:.5f} | "
                  "Avg_Max_Q:{5:2.5f} | Recent reward:{6:.5f}  ".format(episode, frame, count, rall, e, np.mean(average_Q), np.mean(recent_rlist)))
            if epoch_on:
                epoch += 1
                plot_data(epoch, epoch_score, average_reward, epoch_Q, average_Q, mainDQN)
                epoch_on = False
                average_reward = deque()
                average_Q = deque()

if __name__ == "__main__":
    main()