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
        self.wellbeing_fx = 'variance' # absolute, variance, aspiration

        #TASK game has to match rllib's and set alpha appropriately
        #elibility trace
        self.gamma = DQNSetting.GAMMA ##
        self.eligibility_trace = 0
        self.el_alpha=0.5
        
        #defection in M time
        self.context_memory = 1000
        self.defectingDeque = deque(maxlen=self.context_memory)
        

        
        
        self.R = 13 
        self.T = GameSetting.TAGGED_TIME - GameSetting.APPLE_RESPAWN_TIME/self.R
        self.S=0
        self.P =0 
        #A0
        self.aspirational = (self.R + self.T + self.S + self.P)/4
        self.aspiration_beta = 0.2 # aspiration learning rate

        #Core Derivation function
        self.f_u = 1 # conceder 0 < u < 1, linear: u=1, and boulware u>u
        #secondarary emotion derivation g_x
        self.g_v = 1 # 0.2 0.6 1, 2, 3, 5, 10 
    def reset(self):
        self.defectingDeque.clear()
        self.eligibility_trace= 0


        

    def __str__(self):
        return f"agent {self.player_idx}"
 
    def begin_episode(self):
        """Start a new episode"""
        self.action = PlayerAction.STAND_STILL

    def update_internal(self, action, _reward, neigbors, iter_time):
        
        self.update_eligibility(_reward)
        self.update_defection(action, iter_time)
        in_reward = self.emotional_derivation(_reward, neigbors)

        return in_reward

        

    
    def update_eligibility(self, _reward):
        self.eligibility_trace = self.gamma*self.el_alpha*self.eligibility_trace + _reward
    
    def update_defection(self, _action, _iter_time):
        # consider only defection in last M steps

        # decection detected
        if _action == PlayerAction.USE_BEAM:
            self.defectingDeque.append(_iter_time)
        # forgot a detection
        if self.defection_n > 0:
            if  self.defectingDeque[0]+self.context_memory < _iter_time:
                self.defectingDeque.popleft()
    @property
    def defection_n(self):
        return len(self.defectingDeque)

    def social_fairness_context(self, neightbors):
        #cn = 1/N sum((ni_c-ni_d/M)
        if len(neightbors) == 0: return 0
        Cn=0
        for agent in neightbors:
            cooperating_rate = agent.context_memory-2*agent.defection_n
            Cn += cooperating_rate/agent.context_memory
        Cn /=len(neightbors)

        return Cn;
        
    
    def social_fairness_appraisal(self, neightbors):
        context = self.social_fairness_context(neightbors)
        cooperating_rate = self.context_memory-2*self.defection_n
        F = context * (cooperating_rate)/self.context_memory #F = cn x (cn-nd)/M
        # print(context, cooperating_rate)
        
        return F, cooperating_rate < 0 and context >0,  cooperating_rate > 0 and context < 0

    def wellbeing_appraisal(self, _reward):
        #social dilemma payoff
        T = self.T #tempetation
        S = self.S #sucker
        W = 0
        try:
            if self.wellbeing_fx  == 'variance':
                W = ((self.gamma*self.el_alpha*self.eligibility_trace + _reward) - self.eligibility_trace)/self.context_memory
            elif self.wellbeing_fx == 'aspiration':
                h = 1
                W = np.tanh(h*((self.eligibility_trace/self.context_memory) - self.aspirational))
                self.aspirational = (1-self.aspiration_beta)*self.aspirational + self.aspiration_beta*(self.eligibility_trace/self.context_memory)
            else:
                print("the wellbeing function not known")
                sys.exit(1)
        except Exception as err:
            print(err)
            sys.exit(1)

        return W
                
    def emotional_derivation(self, _reward, neighbors):
        wellbeing_appraisal = self.wellbeing_appraisal(_reward)
        fairness_appraisal, exploiting, manipulated = self.social_fairness_appraisal(neighbors)


        E_joy = 0
        E_sad = 0
        E_anger = 0
        E_fearful = 0

        if self.core=='fw':
            # print("Using FW")
            if fairness_appraisal>0:
                if wellbeing_appraisal>0:
                    E_joy = self.core_f(fairness_appraisal)*self.secondary_g(wellbeing_appraisal)
            elif fairness_appraisal<0:
                #agents either defect more in a cooperative environement
                if exploiting:
                    E_fearful = -(self.core_f(-1*fairness_appraisal)*self.secondary_g(wellbeing_appraisal))
                elif manipulated:
                    E_anger = -(self.core_f(-1*fairness_appraisal)*self.secondary_g(-1*wellbeing_appraisal))

        elif self.core == 'wf':
            # print("Using WF")
            if wellbeing_appraisal >0:
                E_joy = self.core_f(wellbeing_appraisal)*self.secondary_g(fairness_appraisal)
            elif wellbeing_appraisal <0:
                E_sad = -(self.core_f(-1*wellbeing_appraisal)*self.core_f(fairness_appraisal))
        emotions = [E_joy>0, E_sad<0, E_anger<0, E_fearful<0]
        # print(emotions)
        assert sum(emotions) <2, "detected more than one emotions"

        return E_joy+E_sad+E_fearful+E_anger

    # Core Derivation Function monotonically maps the desireability of emotion to [0, 1]
    def core_f(self, D_x):
        return D_x**self.f_u

    
    # secondary emotional derivation that maps emotional Intensity [-1, 1] to value [0-1]
    def secondary_g(self, I_x):
        return ((I_x + 1)/2)**self.g_v