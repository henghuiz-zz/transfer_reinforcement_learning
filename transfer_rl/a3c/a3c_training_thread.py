# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import time
import sys

# from async_deep_reinforce.game_state import GameState
from transfer_rl.environment import GymEnvironment
from transfer_rl.a3c.game_ac_network import GameACFFNetwork

from transfer_rl.a3c.constants import ACTION_SIZE, GAMMA, LOCAL_T_MAX,\
    ENTROPY_BETA

LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000


class A3CTrainingThread(object):
    def __init__(self, thread_index, global_network, local_network,
                 initial_learning_rate, learning_rate_input,
                 grad_applier, max_global_time_step, device,
                 rom_names):

        self.thread_index = thread_index
        self.learning_rate_input = learning_rate_input
        self.max_global_time_step = max_global_time_step

        self.local_network = GameACFFNetwork(ACTION_SIZE, thread_index,
                                             local_network, device)

        self.local_network.prepare_loss(ENTROPY_BETA)

        with tf.device(device):
            var_refs = [v._ref() for v in self.local_network.get_vars()]
            self.gradients = tf.gradients(
                self.local_network.total_loss, var_refs,
                gate_gradients=False,
                aggregation_method=None,
                colocate_gradients_with_ops=False)

        self.apply_gradients = grad_applier.apply_gradients(
            global_network.get_vars(),
            self.gradients)

        self.sync = self.local_network.sync_from(global_network)

        self.game_state = GymEnvironment(rom_names, True)

        self.local_t = 0

        self.initial_learning_rate = initial_learning_rate

        self.episode_reward = 0

        # variable controling log output
        self.prev_local_t = 0

    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * (
            self.max_global_time_step - global_time_step) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    def choose_action(self, pi_values):
        return np.random.choice(range(len(pi_values)), p=pi_values)

    def _record_score(self, sess, summary_writer, summary_op,
                      score_input, score, global_t):
        summary_str = sess.run(summary_op, feed_dict={
            score_input: score
        })
        summary_writer.add_summary(summary_str, global_t)
        summary_writer.flush()

    def set_start_time(self, start_time):
        self.start_time = start_time

    def process(self, sess, global_t, summary_writer, summary_op, score_input):
        states = []
        actions = []
        rewards = []
        values = []

        terminal_end = False

        # copy weights from shared to local
        sess.run(self.sync)

        start_local_t = self.local_t

        # t_max times loop
        for i in range(LOCAL_T_MAX):
            state_inc = self.game_state.get_image()

            pi_, value_ = self.local_network.run_policy_and_value(sess, state_inc)
            action = self.choose_action(pi_)

            states.append(state_inc)
            actions.append(action)
            values.append(value_)

            # process game
            reward, terminal = self.game_state.do_action(action)
            self.episode_reward += reward

            # clip reward
            rewards.append(np.clip(reward, -1, 1))

            self.local_t += 1

            if terminal:
                terminal_end = True
                self._record_score(sess, summary_writer, summary_op,
                                   score_input,
                                   self.episode_reward, global_t)

                self.episode_reward = 0
                self.game_state.reset()
                break

        R = 0.0
        if not terminal_end:
            R = self.local_network.run_value(sess, self.game_state.get_image())

        actions.reverse()
        states.reverse()
        rewards.reverse()
        values.reverse()

        batch_si = []
        batch_a = []
        batch_td = []
        batch_R = []

        # compute and accmulate gradients
        for (ai, ri, si, Vi) in zip(actions, rewards, states, values):
            R = ri + GAMMA * R
            td = R - Vi
            a = np.zeros([ACTION_SIZE])
            a[ai] = 1

            batch_si.append(si)
            batch_a.append(a)
            batch_td.append(td)
            batch_R.append(R)

        cur_learning_rate = self._anneal_learning_rate(global_t)

        sess.run(self.apply_gradients,
                 feed_dict={
                     self.local_network.s: batch_si,
                     self.local_network.a: batch_a,
                     self.local_network.td: batch_td,
                     self.local_network.r: batch_R,
                     self.learning_rate_input: cur_learning_rate})

        if (self.thread_index == 0) and (
                self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
            self.prev_local_t += PERFORMANCE_LOG_INTERVAL
            elapsed_time = time.time() - self.start_time
            steps_per_sec = global_t / elapsed_time
            print("### Performance : {} STEPS in {:.0f} sec. "
                  "{:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
                global_t, elapsed_time, steps_per_sec,
                steps_per_sec * 3600 / 1000000.))

        # return advanced local step size
        diff_local_t = self.local_t - start_local_t
        return diff_local_t
