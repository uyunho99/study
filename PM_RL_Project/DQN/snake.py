import gym
import numpy as np
import os
import platform

from gym import spaces
from typing import List, Tuple, Dict, Any, Optional


class Snake(gym.Env):
    def __init__(self, grid_size=(12, 12), mode="array"):
        self.body = None
        self.direction_vec = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)])
        self.direction = None
        self.food = None
        self.mode = mode
        self.board_size = np.array(grid_size)

        self.now = 0
        self.last_eat = 0

        if self.mode == 'array':
            # self.observation_space = spaces.Box(low=-1, high=1, shape=(grid_size[0], grid_size[1]), dtype=int)
            self.observation_space = spaces.Box(
                low=-1, high=np.inf, shape=(grid_size[0]*grid_size[1] + 3, ), dtype=int)

        # elif self.mode == 'image':
        #     self.observation_space = spaces.Box(low=0, high=255, shape=(3, grid_size[0], grid_size[1]), dtype=int)

        self.action_space = spaces.Discrete(4)

        self.reset()

    def get_obs(self):
        if self.mode == 'array':
            obs = np.zeros(self.board_size, dtype=int)
            for i, body in enumerate(self.body, start=1):
                obs[body] = i
            return np.concatenate([obs.flatten(), self.food, [self.direction]])

        # elif self.mode == 'image':
        #     obs = np.zeros(self.observation_space.shape, dtype=int)
        #     obs[0][self.food] = 255
        #     for body in self.body:
        #         obs[1][body] = 255
        #     return obs

    def reset(self):
        self.body = [(np.random.randint(1, self.board_size[0]-1),
                      np.random.randint(1, self.board_size[1]-1))]

        vecs = []
        for _ in range(2):
            while True:
                vec = self.direction_vec[np.random.choice(4)]
                pos = np.array(self.body[-1]) + vec
                if np.logical_and(pos >= 0, pos < self.board_size).all() and tuple(pos) not in self.body:
                    vecs.append(vec)
                    self.body.append(tuple(pos))
                    break

        self.direction = np.random.choice(
            [i for i, vec in enumerate(
                self.direction_vec) if (vec != -vecs[0]).all()]
        )
        self.food = self._generate_food()

        self.now = 0
        self.last_eat = 0

        return self.get_obs()

    def next_pos(self, action: Optional[int] = None):
        head = np.array(self.body[0])
        if action is None:
            action = self.direction

        if action == 0:
            new_head = head - np.array([1, 0])
        elif action == 1:
            new_head = head + np.array([0, 1])
        elif action == 2:
            new_head = head + np.array([1, 0])
        elif action == 3:
            new_head = head - np.array([0, 1])
        else:
            raise ValueError("Invalid action")

        self.body.pop()

        grown = False
        if tuple(new_head) == self.food:
            self.food = self._generate_food()
            self.last_eat = self.now
            grown = True

        return new_head, grown

    def step(self, action: int):
        done, info = False, {}
        reward = self.heuristic()
        self.now += 1
        self.direction = action
        new_head, grown = self.next_pos(action)

        if not np.logical_and(new_head >= 0, new_head < self.board_size).all():
            reward = -10
            done = True
            info['msg'] = 'out of bounds'
        elif tuple(new_head) in self.body[1:]:
            reward = -10
            done = True
            info['msg'] = 'body'
        elif self.now - self.last_eat > 2 * self.board_size.sum():
            reward = -10
            done = True
            info['msg'] = 'timeout'
        else:
            self.body.insert(0, tuple(new_head))
            if grown:
                self.body.append(self.body[-1])
                reward = 100

        return self.get_obs(), reward, done, info

    def render(self, mode="human"):
        # if mode == "char":
        #     black_square = chr(9608) * 2
        #     white_square = chr(9617) * 2
        #     # food = chr(9679) * 2
        #     food = chr(9675)
        # else:
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

        render_board = self.get_obs().copy()
        food_pos = render_board[-3:-1]
        render_board = render_board[:-
                                    3].reshape(-1, *self.board_size).squeeze()
        if self.mode == 'image':
            render_board = (render_board[1] - render_board[0])/255
        elif self.mode == 'array':
            render_board[tuple(food_pos)] = -1
        render_board = np.vectorize(encode)(render_board)
        for row in render_board:
            print(''.join(row))

    def heuristic(self):
        head = np.array(self.body[0])
        food = np.array(self.food)
        return -np.linalg.norm(head - food) / self.board_size.max()

    def _generate_food(self):
        while True:
            food = (np.random.randint(0, self.board_size[0]),
                    np.random.randint(0, self.board_size[1]))
            if food not in self.body:
                return food


# env = Snake(grid_size=(9, 9))
# obs = env.reset()
# env.render()
# done, info = False, {}
# cum_reward = 0
# while not done:
#     action = int(input('action: '))
#     obs, reward, done, info = env.step(action)
#     env.render()
#     cum_reward += reward

# print(cum_reward, info)
