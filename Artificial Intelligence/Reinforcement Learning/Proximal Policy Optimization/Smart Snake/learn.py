
from stable_baselines3 import PPO
import os

from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from snakeenv import SnekEnv

render_mode = 'human'
# render_mode = 'none'

models_dir = "models"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


def make_env(_render_mode):
    """
    Utility function for multiprocessed env.

    :param _render_mode:
    :param num_env: (int) the number of environments you wish to have in subprocesses
    """

    def _init():
        _env = SnekEnv(_render_mode)
        _env.reset()
        return _env

    return _init


# this is the main process
if __name__ == '__main__':
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    env = VecMonitor(SubprocVecEnv([make_env(render_mode) for i in range(num_cpu)]), "TestMonitor")

    # model = PPO('MlpPolicy', env, verbose=0, learning_rate=0.0003, ent_coef=0.01, tensorboard_log=logdir)
    model = PPO.load("models/436.zip", env=env)

    print("------------- Start Learning --------------------------------")
    for i in range(1, 10000):
        model.learn(total_timesteps=10000, reset_num_timesteps=False, tb_log_name="PPO-0003-state_change-200_mem")

        if render_mode == 'none' and i%4 == 0:
            model.save(f"{models_dir}/{i}")
    print("------------- Done Learning ---------------------------------")

    # obs, info = env.reset()
    #
    # for _ in range(1,10):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()

    env.close()



# env = SnekEnv(render_mode)
# env.reset()
#
# model_path = f'{models_dir}/470.zip'
# model = PPO.load(model_path, env=env)
#
# # model = PPO('MlpPolicy', env, verbose=1, ent_coef=0.01, tensorboard_log=logdir)
#
# TIMESTEPS = 10000
#
# for i in range(1, 10000000):
#     model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False, tb_log_name="PPO")
#
#     if render_mode == 'none' and i % 10 == 0:
#         model.save(f"{models_dir}/{i}")
#
# env.close()
