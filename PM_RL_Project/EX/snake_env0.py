import gym
import numpy as np

from gym import spaces
from typing import List, Tuple, Dict, Any, Optional, Union

from stable_baselines3.common.env_checker import check_env

import platform
import os

__version__ = "0.0.0"


class Snake(gym.Env):
    def __init__(self,
                 grid_size=(12, 12)):

        self.__version__ = __version__
        self.body = [(0, 0)]
        self.direction_vec = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)])
        self.direction = 0
        self.food = (0, 0)
        self.board = np.zeros(grid_size, dtype=np.uint8)
        self.board_shape = np.array(grid_size)

        self.now = 0
        self.last_eat = 0
        self.max_time = 4 * self.board_shape.sum()

        # TODO: define observation space
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(grid_size[0]*grid_size[1]+4), dtype=int)

        self.action_space = spaces.Discrete(4)
        self.reset()

    def get_obs(self):
        # TODO: return observation
        candidates = self.body[0] + self.direction_vec
        indices = np.arange(4)
        bound_condition = np.logical_and((candidates >= 0).all(
            axis=1), (candidates < self.board_shape).all(axis=1))
        indices = indices[bound_condition]
        candidates = candidates[bound_condition]
        body_condition = self.board[candidates[:, 0], candidates[:, 1]] == 0
        indices = indices[body_condition]
        action_mask = np.zeros(4, dtype=np.uint8)
        action_mask[indices] = 1

        obs = [self.food]
        obs.extend(self.body)
        obs.extend([(-1, -1)] * (self.board.shape[0] *
                   self.board.shape[1] - len(self.body) - 1))
        obs = np.array(obs) + 1
        obs = np.concatenate([obs.flatten(), action_mask])
        return obs

    def reset(self):
        self.board = np.zeros(self.board_shape, dtype=np.uint8)
        self.body = None  # TODO: reset body, check board filled correctly
        # [(np.random.randint(1, self.board_size[0]-1),
        #   np.random.randint(1, self.board_size[1]-1))]
        self.direction = 0

        self.food = self._generate_food()

        self.now = 0
        self.last_eat = 0

        return self.get_obs()

    def render(self, mode="human"):
        if mode == "char":
            black_square = chr(9608) * 2
            white_square = chr(9617) * 2
            # food = chr(9679) * 2
            food = chr(9675) * 2
        else:
            black_square = chr(int('2b1b', 16))
            white_square = chr(int('2b1c', 16))
            food = chr(int('1f34e', 16))
            # food = chr(int('1f7e7', 16))

        def encode(v):
            if v == 0:
                return white_square
            elif v > 0:
                return black_square
            elif v == -1:
                return food

        if platform.system() == 'Windows':
            os.system('cls')
        else:
            os.system('clear')

        render_board = self.board.astype(int).copy()
        food_pos = self.food
        render_board[food_pos] = -1
        render_board = np.vectorize(encode)(render_board)
        for row in render_board:
            print(''.join(row))

    def step(self, action):
        self.now += 1
        self.direction = action

        done, info = False, {}
        new_head = self.body[0] + self.direction_vec[self.direction]
        self.board[self.body.pop()] -= 1

        bound_condition = (new_head >= 0).all() and (
            new_head < self.board_shape).all()
        body_condition = tuple(new_head) not in self.body
        starve_condition = self.now - self.last_eat <= self.max_time
        if bound_condition and body_condition and starve_condition:
            self.body.insert(0, tuple(new_head))
            self.board[tuple(new_head)] = 1
            if self.food == tuple(new_head):
                reward = 10
                self.last_eat = self.now
                self.body.append(self.body[-1])
                self.board[self.body[-1]] += 1
                self.food = self._generate_food()
            else:
                reward = self.heuristic(
                    self.body, self.food, self.direction) * 0.5
        else:
            done = True
            reward = -10
            if not bound_condition:
                msg = 'out of body'
            elif not body_condition:
                msg = 'body'
            else:
                msg = 'timeout'
            info['mgs'] = msg
        return self.get_obs(), reward, done, info

    def heuristic(self, body, food, direction):
        # TODO: calculate supplement reward if needed
        pass

    def _generate_food(self):
        while True:
            food = (np.random.randint(0, self.board_shape[0]),
                    np.random.randint(0, self.board_shape[1]))
            if food not in self.body:
                # self._record_food.append(food)
                return food


if __name__ == "__main__":
    env = Snake()
    check_env(env)
