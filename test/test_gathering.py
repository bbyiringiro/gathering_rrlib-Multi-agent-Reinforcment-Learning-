
from envv2 import *
from configs import ALL_PLAYER_ACTIONS

if __name__ == '__main__':
    # set_global_seeds(DQNSetting.SEED)
    Game = GatherMultEnv()
    # Game.visual = True
    Game.reset()

    print(Game.observation_space)
    while True:
        actions = {'agent-0':random.choice(ALL_PLAYER_ACTIONS), 'agent-1':random.choice(ALL_PLAYER_ACTIONS)}
        observations, rewards, dones, info =Game.step(actions)
        # print(rewards)
        

