from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy


from snake import Snake

env = Snake(grid_size=(9, 9), mode='array')
eval_env = Snake(grid_size=(9, 9), mode='array')

learn_kwargs = dict(total_timesteps=200000,
                    log_interval=10,
                    eval_env=env,
                    eval_freq=500,
                    n_eval_episodes=10,
                    tb_log_name=f"Snake_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    eval_log_path="./")

model = DQN(MlpPolicy, env, verbose=1, learning_rate=1e-4, tensorboard_log="./snake_tensorboard/",
            exploration_fraction=0.3, batch_size=32, train_freq=1, policy_kwargs={'net_arch': [128, 128]})

print("model structure: ")
print(model.policy)
input("Press Enter to continue...")

model.learn(**learn_kwargs)
