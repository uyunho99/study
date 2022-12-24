import argparse
import os
import platform
import time
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy

from snake import Snake


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best_model')
    parser.add_argument('--max-steps', type=int, default=1000)
    args = parser.parse_known_args()[0]

    print(args)

    env = Snake(grid_size=(9, 9))
    # model = DQN.load(args.model)
    model = DQN.load(args.model)

    obs = env.reset()
    cum_reward = 0
    done, info = False, {}
    i = 0
    while not done and i < args.max_steps:
        time.sleep(0.3)
        if platform.system() == 'Windows':
            os.system('cls')
        else:
            os.system('clear')
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        cum_reward += reward
        env.render()
        i += 1
    print(f"cumulative reward: {cum_reward}")
    print(f"info: {info}")


if __name__ == "__main__":
    main()
