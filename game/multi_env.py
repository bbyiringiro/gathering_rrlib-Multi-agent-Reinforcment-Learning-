from configs import *
import sys
import pygame
import matplotlib.pyplot as plt
import numpy as np
from game.gathering import *
import json
import time
from configs import *

from ray.rllib.env import MultiAgentEnv

from agent.basic import GeneralAgent
from gym.spaces import Box
from gym.spaces import Discrete

from configs import ALL_PLAYER_ACTIONS

class GatherMultEnv(MultiAgentEnv):

    def __init__(self):


        self.fps_clock = None
        self.screen = None
        self.visual =DQNSetting.VISUAL_GUI
        self.env = None
        
        self.agents = {} #[]
        self.timestep_watch = Stopwatch()
        self.logger = Params().logger
        self.episode = None  # number of episode for training
        self.visualize = DQNSetting.VISUAL_DATA


        self.iteration = None
        self.n_steps = DQNSetting.TOTAL_STEPS_PER_EPISODE
        self.n_episodes = DQNSetting.TOTAL_NUM_EPISODE

        self.set_up()
        self.num_agents = len(self.agents)


    def set_up(self):
        self.logger.warning("Command line environment setting up")
        with open('gathering.json') as cfg:
            env_config = json.load(cfg)
        self.env = EnvironmentGathering(env_config)
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
        self.fps_clock = pygame.time.Clock()

        # Main game loop.
        self.iteration += 1  
        observations = {}
        rewards = {}
        dones = {}
        info = {}
        for agent_id, agent in self.agents.items():
            obs, reward, done = self.move(agent,actions[agent_id])
            observations[agent_id] = obs
            rewards[agent_id] = reward
            dones[agent_id] = done
        dones["__all__"] = np.any(list(dones.values()))


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
    def agent_pos(self):
        return {id:self.env.player_list[agent.player_idx].get_position() for id, agent in self.agents.items()}


    @property
    def action_space(self):
        return Discrete(len(ALL_PLAYER_ACTIONS))

    @property
    def observation_space(self):
        return Box(low=0, high=255, shape=(*GameSetting.player_view, 3), dtype=np.int64) #np.int8

