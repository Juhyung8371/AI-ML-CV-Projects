import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor

from snakeenv import SnekEnv

num_cpu = 4  # Number of processes to use

models_dir = "models"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


class ScoreCallBack(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

        # self.log_dir = "logs"  # Change this to the directory where you want to save the logs

    def _on_step(self) -> bool:
        ave_score = 0
        for i in range(num_cpu):
            ave_score += self.locals['infos'][i]['score']
        ave_score /= num_cpu

        # Log scalar value
        self.logger.record("score", ave_score)

        # TODO save best model callback
        return True


def make_env(_render_mode, map_size):
    """
    Utility function for multiprocessed env.

    :param _render_mode:
    """

    def _init():
        _env = SnekEnv(_render_mode, map_size)
        _env.reset()
        return _env

    return _init


# this is the main process
if __name__ == '__main__':

    # Create the vectorized environment
    env_lr_low = VecMonitor(SubprocVecEnv([make_env('none', 10) for i in range(num_cpu)]), "TestMonitor")

    model_lr_low = PPO('MlpPolicy', env_lr_low, verbose=0, batch_size=1024,
                       learning_rate=0.0003, ent_coef=0.001, gamma=0.99,
                       tensorboard_log=logdir)

    print("------------- Start Learning --------------------------------")
    for i in range(1, 10000):
        model_lr_low.learn(total_timesteps=50000, callback=ScoreCallBack(),
                           reset_num_timesteps=False, tb_log_name="__10x10")
        if i % 4 == 0:
            model_lr_low.save(f"{models_dir}/__10x10{i}")

    print("------------- Done Learning ---------------------------------")

    env_lr_low.close()
