# -*- coding: utf-8 -*-
import tensorflow as tf
import threading
import math
import os
import time

from transfer_rl.a3c.game_ac_network import GameACFFNetwork
from transfer_rl.a3c.a3c_training_thread import A3CTrainingThread
from transfer_rl.a3c.rmsprop_applier import RMSPropApplier
from transfer_rl.a3c.constants import *
from transfer_rl.networks import CNNTypeOne


def log_uniform(lo, hi, rate):
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1 - rate) + log_hi * rate
    return math.exp(v)


def train_function(parallel_index):
    global global_t
    global STOP_STEP

    training_thread = training_threads[parallel_index]
    # set start_time
    start_time = time.time() - wall_t
    training_thread.set_start_time(start_time)

    while True:
        if global_t > STOP_STEP:
            break

        diff_global_t = training_thread.process(sess, global_t, summary_writer,
                                                summary_op, score_input)
        global_t += diff_global_t


if __name__ == '__main__':
    ROM = 'Breakout-v0'

    CHECKPOINT_DIR = '../save/teacher/' + ROM

    if not os.path.isdir(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    device = "/gpu:0"

    initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW,
                                        INITIAL_ALPHA_HIGH,
                                        INITIAL_ALPHA_LOG_RATE)

    global_t = 0

    global_network_core = CNNTypeOne(ROM, 6, True, prefix='student_')
    global_network = GameACFFNetwork(
        ACTION_SIZE, -1, global_network_core, device)

    all_network_cores = []
    training_threads = []

    learning_rate_input = tf.placeholder("float")

    grad_applier = RMSPropApplier(learning_rate=learning_rate_input,
                                  decay=RMSP_ALPHA,
                                  momentum=0.0,
                                  epsilon=RMSP_EPSILON,
                                  clip_norm=GRAD_NORM_CLIP,
                                  device=device)

    for i in range(PARALLEL_SIZE):
        network_core = CNNTypeOne(ROM, 6, True, prefix='a3c'+str(i))
        training_thread = A3CTrainingThread(i, global_network, network_core,
                                            initial_learning_rate,
                                            learning_rate_input,
                                            grad_applier, MAX_TIME_STEP,
                                            device=device, rom_names=ROM)
        all_network_cores.append(network_core)
        training_threads.append(training_thread)

    # prepare session
    tfconfig = tf.ConfigProto(log_device_placement=False,
                              allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    init = tf.global_variables_initializer()
    sess.run(init)

    # summary for tensorboard
    score_input = tf.placeholder(tf.int32)
    tf.summary.scalar("score", score_input)

    # init or load checkpoint with saver

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(CHECKPOINT_DIR, sess.graph)

    wall_t = 0.0

    # set start time
    start_time = time.time() - wall_t
    global STOP_STEP
    STOP_STEP = global_t
    while True:
        STOP_STEP += SAVE_STEP
        train_threads = []
        for i in range(PARALLEL_SIZE):
            train_threads.append(
                threading.Thread(target=train_function, args=(i,)))

        for t in train_threads:
            t.start()

        for t in train_threads:
            t.join()

        print('Now saving data. Please wait')

        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)

        # write wall time
        global_network_core.save_variable(
            sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_t)
        print('Data saved')