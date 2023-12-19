import os

from stable_baselines3 import PPO

from snakeenv_cnn import SnekEnv

GAME_LENGTH_THRES = 4000
GAMES_N = 2

_env = SnekEnv('human', 10)
_env.reset()
model = PPO.load("models/snakeenv_final_model.zip", _env=_env)
model.ent_coef = 0  # no need for random exploration in test

complete_game_len = []
complete_game_score = []
incomplete_game_len = []
incomplete_game_score = []
stuck_game_len = []
stuck_game_score = []


def main():
    footage_dir = 'final_footage'
    if not os.path.exists(footage_dir):
        os.makedirs(footage_dir)

    for i in range(GAMES_N):

        steps = 0
        obs, info = _env.reset()
        done = False

        stuck = False

        # un-comment some lines to save the game footage
        # imgs = deque([], maxlen=2000)

        while not done:
            action, state = model.predict(obs)
            obs, reward, done, trunc, info = _env.step(action)

            steps += 1

            # if info['score'] > -1:
            #     imgs.append(_env.get_screen_img())

            # force stop the game
            if steps == GAME_LENGTH_THRES and not done:
                done = True
                stuck = True

        score = info['score']

        if score == 97:
            complete_game_score.append(score)
            complete_game_len.append(steps)
        elif stuck:
            stuck_game_score.append(score)
            stuck_game_len.append(steps)
        else:
            incomplete_game_score.append(score)
            incomplete_game_len.append(steps)

        print(f"Game: {i}, Total Steps: {steps}, score: {score}")

        # for i, img in enumerate(imgs):
        #     jpeg = Image.frombytes('RGB', (10*20, 10*20), img)
        #     jpeg.save(f'{footage_dir}/frame{i}.png')

    complete_game_num = len(complete_game_score)
    incomplete_game_num = len(incomplete_game_score)
    stuck_game_num = len(stuck_game_score)

    complete_game_ave_score = sum(complete_game_score) / complete_game_num
    complete_game_ave_len = sum(complete_game_len) / complete_game_num

    incomplete_game_ave_score = sum(incomplete_game_score) / incomplete_game_num
    incomplete_game_ave_len = sum(incomplete_game_len) / incomplete_game_num

    stuck_game_ave_score = sum(stuck_game_score) / stuck_game_num
    stuck_game_ave_len = sum(stuck_game_len) / stuck_game_num

    print(f'Complete Game# {complete_game_num}, score: {complete_game_ave_score}, length: {complete_game_ave_len}')
    print(
        f'Incomplete Game# {incomplete_game_num}, score: {incomplete_game_ave_score}, length: {incomplete_game_ave_len}')
    print(f'Stuck Game# {stuck_game_num}, score: {stuck_game_ave_score}, length: {stuck_game_ave_len}')

    _env.close()


if __name__ == "__main__":
    main()
