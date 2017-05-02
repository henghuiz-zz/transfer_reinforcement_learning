import gym
import numpy as np
from collections import deque
import cv2


history_play = 4


class GymEnvironment:
    def __init__(self, game_name, resize=True):
        self.game_name = game_name
        self.env = gym.make(game_name)
        self.num_action = self.env.action_space.n
        self.history = deque(maxlen=history_play)
        self.resize = resize
        self.reset()

    def reset(self):
        observation = self.env.reset()
        if self.resize:
            observation = cv2.resize(observation, (84, 84))
        observation = observation / 255.0
        self.history.append(observation)

    def get_image(self):
        diff_len = history_play - len(self.history)
        if diff_len == 0:
            state = np.concatenate(self.history, axis=2)
        else:
            zeros = [np.zeros_like(self.history[0]) for k in range(diff_len)]
            for k in self.history:
                zeros.append(k)
            assert len(zeros) == self.history.maxlen
            state = np.concatenate(zeros, axis=2)
        return state

    def do_action(self, action):
        observation, reward, done, _ = self.env.step(action)
        if done:
            self.reset()
        else:
            if self.resize:
                observation = cv2.resize(observation, (84, 84))
            observation = observation / 255.0
            self.history.append(observation)
        return reward, done
