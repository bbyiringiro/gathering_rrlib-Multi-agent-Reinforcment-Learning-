from collections import deque
from configs import PlayerAction
from configs import GameSetting, DQNSetting
import sys



from configs import ALL_PLAYER_ACTIONS
import numpy as np


class GeneralAgent(object):
    """The agent controller by human"""

    def __init__(self):
        super(GeneralAgent, self).__init__()
        self.player_idx = None
        self.step = None  # total number of steps in all training episode
        self.step_in_an_episode = None  # number of steps in an episode
        self.action = None  # current action

        self.core = 'fw' # fw or wf (fairness then wellbing, or wellbeing then fairness)
        self.wellbeing_fx = 'variance' # variance, aspiration

        self.episode_len = 1000

        #elibility trace
        self.fairness_gamma = .99
        self.fairness_alpha = 1
        self.fairness_epsilon = 0.1

        self.prosocial_metric = 0

        self.reward_gamma = .99
        self.reward_alpha=1

        self.smoothen_wellbeing = 0
        


        self.aspirational = 0.5 
        self.aspiration_beta = 0.5 # aspiration learning rate
        

        #Core Derivation function
        self.f_u = 1 # conceder 0 < u < 1, linear: u=1, and boulware u>u
        #secondarary emotion derivation g_x
        self.g_v = 1 # 0.2 0.6 1, 2, 3, 5, 10 


    def reset(self):
        self.smoothen_wellbeing= 0
        self.prosocial_metric = 0


        

    def __str__(self):
        return f"agent {self.player_idx}"
 
    def begin_episode(self):
        """Start a new episode"""
        self.action = PlayerAction.STAND_STILL

    def update_internal(self, action, _reward, neigbors, current_iter, all_alive=False):
        # update prosocial metrics
        if action == PlayerAction.USE_BEAM and all_alive:
            self.update_prosocial(1)
        else:
            self.update_prosocial(0)
        

        
        in_reward = self.emotional_derivation(_reward, neigbors, current_iter)

        # update wellbeing
        self.update_wellbeing(_reward)
        return in_reward

        

    
    def update_wellbeing(self, _reward):
        self.smoothen_wellbeing = self.reward_gamma*self.reward_alpha*self.smoothen_wellbeing + _reward

    def update_prosocial(self, happened):
        self.prosocial_metric = self.fairness_gamma*self.fairness_alpha*self.prosocial_metric  +  happened

    def social_fairness_appraisal(self, neightbors):
        if len(neightbors) == 0: return 0
        Cn=0
        temp_sum = self.prosocial_metric 
        for agent in neightbors:
            temp_sum += agent.prosocial_metric
            Cn += agent.prosocial_metric - self.prosocial_metric 
        return 0 if temp_sum == 0 else Cn/temp_sum

    def wellbeing_appraisal(self, _reward, current_iter):
        #social dilemma payoff
        # T = self.T #tempetation
        # S = self.S #sucker
        
        W = 0
        if self.wellbeing_fx  == 'variance':
            W = ((self.reward_gamma*self.reward_alpha*self.smoothen_wellbeing + _reward) - self.smoothen_wellbeing)/((current_iter-self.reward_gamma*current_iter)+1)
            # W = ((self.smoothen_wellbeing + _reward) - self.smoothen_wellbeing)
        elif self.wellbeing_fx == 'aspiration':
            h = 1
            W = np.tanh(h*((self.smoothen_wellbeing/current_iter) - self.aspirational))
            self.aspirational = (1-self.aspiration_beta)*self.aspirational + self.aspiration_beta*(self.smoothen_wellbeing/(current_iter+1))
        # print(self.smoothen_wellbeing)
        # if W > 0:
        #     print(W)
        #     # sys.exit(0)
        #     import time
        #     time.sleep(1)
        
        # W = np.clip(W, -1, 1) # clip due to floating points error
        return  W
                
    def emotional_derivation(self, _reward, neighbors, current_iter):
        wellbeing_appraisal = self.wellbeing_appraisal(_reward, current_iter)
        fairness_appraisal = self.social_fairness_appraisal(neighbors)


        
        

        # print("wellbeing: ",wellbeing_appraisal)
        # print("fairness appraisal: ",fairness_appraisal)
        assert(abs(wellbeing_appraisal) <=1)
        assert(abs(fairness_appraisal) <=1)

        if len(neighbors) == 0:
            return wellbeing_appraisal, False, False, False, False
        



        E_joy = 0
        E_sad = 0
        E_anger = 0
        E_fearful = 0

        

        if self.core=='fw':
            # print("Using FW")
            if np.abs(fairness_appraisal) <=self.fairness_epsilon:
                # if wellbeing_appraisal>0:
                F = (self.fairness_epsilon-np.abs(fairness_appraisal))/self.fairness_epsilon
                    
                E_joy = self.core_f(F) * self.secondary_g(wellbeing_appraisal)
            elif fairness_appraisal>0: #exploiting
                E_fearful = -self.core_f(abs(fairness_appraisal))*self.secondary_g(wellbeing_appraisal)
                
            else: ## same lines but be useful for stats
                E_anger = -self.core_f(abs(fairness_appraisal))*self.secondary_g(-1.0*wellbeing_appraisal)

        elif self.core == 'wf':
            # print("Using WF")
            if np.abs(fairness_appraisal) <=self.fairness_epsilon:
                F = (self.fairness_epsilon-np.abs(fairness_appraisal))/self.fairness_epsilon
            else:
                F =  -1.*abs(fairness_appraisal)
            
            if wellbeing_appraisal >0:
                E_joy = self.core_f(wellbeing_appraisal)*self.secondary_g(F)
            elif wellbeing_appraisal <0:
                E_sad = -(self.core_f(-1*wellbeing_appraisal)*self.secondary_g(F))
        emotions = [E_joy>0, E_sad<0, E_anger<0, E_fearful<0]
        # print(emotions)
        assert sum(emotions) <2, "detected more than one emotions"

        # print(f"joy {E_joy} + sad {E_sad} + fear {E_fearful} + anger{E_anger}")

        assert(E_fearful <= 0 and E_anger <=0 and E_sad<=0 and E_joy >=0)

        return E_joy + E_sad + E_fearful + E_anger, E_joy>0, E_sad<0, E_fearful<0, E_anger<0, 

    # Core Derivation Function monotonically maps the desireability of emotion to [0, 1]
    def core_f(self, D_x):
        return D_x**self.f_u

    
    # secondary emotional derivation that maps emotional Intensity [-1, 1] to value [0-1]
    def secondary_g(self, I_x):
        return ((I_x + 1)/2)**self.g_v
