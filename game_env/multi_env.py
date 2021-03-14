from configs import *
import sys
import pygame
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

from configs import ALL_PLAYER_ACTIONS

class GatherMultEnv(MultiAgentEnv):

    def __init__(self, env_config):


        self.fps_clock = None
        self.screen = None
        self.visual = env_config['visual']
        self.env = None
        
        self.agents = {} #[]
        self.timestep_watch = Stopwatch()
        self.logger = Params().logger
        self.episode = None  # number of episode for training
        self.visualize = DQNSetting.VISUAL_DATA


        self.iteration = 0
        self.n_steps = DQNSetting.TOTAL_STEPS_PER_EPISODE
        self.n_episodes = DQNSetting.TOTAL_NUM_EPISODE
        if not env_config['init']:
            self.set_up(env_config)
            #IMRL
            self.full_observable = env_config['full_obs']
            self.intrinsically_motivated = env_config['imrl']
            print(f" imrl {env_config['imrl']} ")
        self.num_agents = len(self.agents)

        

    def set_up(self, config):
        self.logger.warning("Command line environment setting up")
        with open('gathering.json') as cfg:
            env_config = json.load(cfg)

        self.logger.info(f"f n-time{config['n_tag']} n-apple {config['n_apple']}")
        self.env = EnvironmentGathering(env_config, config['n_tag'], config['n_apple'])
        self.logger.warning("Loading map successfully")
        self.start_time = time.time()
        if self.visual:
            pygame.init()
            screen_size = (self.env.grid.width * GameSetting.CELL_SIZE, self.env.grid.height * GameSetting.CELL_SIZE)
            self.screen = pygame.display.set_mode(screen_size)
            self.screen.fill(Colors.SCREEN_BACKGROUND)
            pygame.display.set_caption('Gathering')
        self.iteration = 0

        """ Load the RL agent into the Game GUI. """
        self.agents = {'agent-0':GeneralAgent(), 'agent-1':GeneralAgent()}
        idx = 0
        for agent in self.agents.values():
            agent.player_idx = idx
            idx += 1


    def reset(self):
        self.timestep_watch.reset()
        self.env.new_episode()
        self.iteration = 0
        obs = {}
        for agent_id, agent in self.agents.items():
            agent.begin_episode()
            obs[agent_id]= self.env.player_list[agent.player_idx].convert_observation_to_rgb().reshape(-1, 1)
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
        if self.visual:
            self.fps_clock = pygame.time.Clock()

        # Main game loop.
        
        # print(self.iteration) 
        observations = {}
        rewards = {}
        dones = {}
        # info = {'agent-0':np.array([0]), 'agent-1':np.array([0])}
        info={}
        

        aggressivenss = 0
        for agent_id, agent in self.agents.items():
            obs, reward, done = self.move(agent,actions[agent_id])
            observations[agent_id] = obs.reshape(-1,1)
            rewards[agent_id] = reward
            dones[agent_id] = done
            info[agent_id] = {'exR':reward}
            info[agent_id]['agent_action']= actions[agent_id]
            if self.intrinsically_motivated:
                in_reward = agent.update_internal(actions[agent_id],\
                reward,\
                self.get_neigbors(agent_id, self.env.player_list[agent.player_idx].observable_view),\
                self.iteration)
                #TASK separated the metrics collection for intrinsic vs extrinsic
                rewards[agent_id] += in_reward
                info[agent_id]['inR'] = in_reward
                # print(in_reward)
            else:
                info[agent_id]['inR'] = 0
            
                
        dones["__all__"] = np.any(list(dones.values()))

        self.iteration += 1

        #add aggressivness to the info report



        if self.visual:
            self.draw_all_cells()
            pygame.display.update()
            self.fps_clock.tick(GameSetting.FPS_LIMIT) 
        return observations, rewards, dones, info




    def draw_a_cell(self, x, y):
        """ Draw the cell specified by the field coordinates. """
        cell_coords = pygame.Rect(
            x * GameSetting.CELL_SIZE,
            y * GameSetting.CELL_SIZE,
            GameSetting.CELL_SIZE,
            GameSetting.CELL_SIZE,
        )
        if self.env.view_array[y,x] == CellType.EMPTY:
            pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, cell_coords)
        else:
            color = Colors.CELL_TYPE[self.env.view_array[y, x]]
            pygame.draw.rect(self.screen, color, cell_coords)

    def draw_all_cells(self):
        """ Draw the entire game frame. """
        for x in range(self.env.grid.width):
            for y in range(self.env.grid.height):
                self.draw_a_cell(x, y)
    @property
    def all_agents_pos(self):
        return {id:self.env.player_list[agent.player_idx].get_position() for id, agent in self.agents.items()}


    @property
    def action_space(self):
        return Discrete(len(ALL_PLAYER_ACTIONS))

    @property
    def observation_space(self):
        return Box(low=0, high=255, shape=(np.product(GameSetting.player_view)*3,1), dtype=np.uint8) #np.int8

    
    
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

