from configs import *
import sys,os
sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
from game_env.gathering import *
import json
import time
from configs import *

from ray.rllib.env import MultiAgentEnv

from agent.basic import GeneralAgent
from gym.spaces import Box
from gym.spaces import Discrete
from utils.utility import convert_observation_to_rgb


from configs import ALL_PLAYER_ACTIONS
print(DQNSetting.TOTAL_STEPS_PER_EPISODE)

class GatherMultEnv(MultiAgentEnv):

    def __init__(self):


        self.screen = None
        self.visual =True
        self.env = None
        
        self.agents = {} #[]
        self.timestep_watch = Stopwatch()
        self.logger = Params().logger
        self.episode = None  # number of episode for training
        self.visualize = True


        self.iteration = None
        self.n_steps = DQNSetting.TOTAL_STEPS_PER_EPISODE
        self.n_episodes = DQNSetting.TOTAL_NUM_EPISODE

        self.set_up()
        self.num_agents = len(self.agents)
        self.fig, self.ax = plt.subplots()
        plt.ion()

        self.full_observable = True




    def set_up(self):
        self.logger.warning("Command line environment setting up")
        with open('gathering.json') as cfg:
            env_config = json.load(cfg)
        self.env = EnvironmentGathering(env_config, 6, 60)
        self.logger.warning("Loading map successfully")
        self.start_time = time.time()

        self.iteration = 0

        """ Load the RL agent into the Game GUI. """
        self.agents = {'agent-0':GeneralAgent(), 'agent-1':GeneralAgent()}
        idx = 0
        for agent in self.agents.values():
            agent.player_idx = idx
            idx += 1

    def show(self, img):
        plt.imshow(img, interpolation='nearest')
        plt.draw()
        # plt.savefig('frames/'+str(self.iteration)+'.pdf')
        plt.pause(.0001)


    def reset(self):
        self.timestep_watch.reset()
        self.env.new_episode()
        self.iteration = 0
        obs = {}
        for agent_id, agent in self.agents.items():
            agent.begin_episode()
            obs[agent_id]= self.env.player_list[agent.player_idx].convert_observation_to_rgb()
        return obs
        

    def move(self, agent, action):
        agent.step_in_an_episode = self.iteration
        observation = self.env.player_list[agent.player_idx].convert_observation_to_rgb()
        # agent.action = agent.act(observation)

        self.env.take_action(action, self.env.player_list[agent.player_idx]) #changes agent state
        self.env.move(agent.step_in_an_episode) # changes env state
        cur_reward = self.env.player_list[agent.player_idx].reward
        return observation, cur_reward, False

    def step(self, actions):
        # Main game loop.
        self.iteration += 1  
        if self.iteration < 1000:
            self.iteration += 0
        observations = {}
        rewards = {}
        dones = {}
        info = {}
        calculate_aggress = False
        if not self.env.player_list[0].is_tagged and not self.env.player_list[1].is_tagged:
            calculate_aggress = True
        for agent_id, agent in self.agents.items():
            obs, reward, done= self.move(agent,actions[agent_id])
            observations[agent_id] = obs
            print(obs.reshape(-1,1).shape)
            rewards[agent_id] = reward
            in_reward, joy, sad, fear, anger = agent.update_internal(actions[agent_id],\
                reward,\
                self.get_neigbors(agent_id, self.env.player_list[agent.player_idx].observable_view),\
                self.iteration, calculate_aggress)
            print(f' InR: {in_reward}')
            # if in_reward <0:
            #     import time
            #     time.sleep(5)
            
            # print(f'elibility trace {agent_id} {agent.eligibility_trace}')
            # print(f'defections n{agent_id} {agent.defection_n}')
            dones[agent_id] = done
        dones["__all__"] = np.any(list(dones.values()))


        # full_obs_rgb = convert_observation_to_rgb(self.env.view_array, self.env.view_array, )
        # self.show(observations['agent-0'])
        # if in_reward < 0:
        #     import sys
        #     sys.exit()
        return observations, rewards, dones, info
    
    def get_neigbors(self, agent_id, agent_view):
        if self.full_observable:
            return [neigbor for id, neigbor in self.agents.items() if id != agent_id ]
        else:
            return [neigbor for id, neigbor in self.agents.items() if id != agent_id and \
                self.is_neibor_in_view(agent_view,self.env.player_list[neigbor.player_idx].get_position())]
    def is_neibor_in_view(self, agent_view, agent2_pos):
        # agent_view = x_min, x_max, y_min, y_max 
        return agent2_pos.x >= agent_view[0] and agent2_pos.x <= agent_view[1]\
            and agent2_pos.y >=agent_view[2] and agent2_pos.y <= agent_view[3]
        
 
    def agent_pos(self, agent):
        return self.env.player_list[agent.player_idx].get_position()
    



    @property
    def action_space(self):
        return Discrete(len(ALL_PLAYER_ACTIONS))

    @property
    def observation_space(self):
        return Box(low=0, high=255, shape=(*GameSetting.player_view, 3), dtype=np.uint8) 
        

