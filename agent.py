import numpy as np
import random
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 0.005
        self.alpha = 0.06
        self.gamma = 0.98
        #self.Q = sarsa(env, 5000, .01)
        
    def update_Q(self,Qsa, Qsa_next, reward, alpha, gamma):
        """ updates the action-value function estimate using the most recent time step """
        return Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))

    def epsilon_greedy_probs(self, Q_s):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        
        #epsilon = 1.0 / i_episode
        #if eps is not None:
        #    epsilon = eps
        #print('epsilon ',epsilon, self.epsilon)
        policy_s = np.ones(self.nA) * self.epsilon / self.nA
        policy_s[np.argmax(Q_s)] = 1 - self.epsilon + (self.epsilon / self.nA)
        #print('policy ',policy_s)
        return policy_s
    
    
    

    def select_action(self, state):
        """ Given the state, select an action.
        Params
        ======
        - state: the current state of the environment
        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        act_space = [i for i in range(0, self.nA)]
        action = np.random.choice(np.arange(self.nA),
                                  p = self.epsilon_greedy_probs(self.Q[state])) if state in self.Q else random.choice(act_space)
        # return np.random.choice(self.nA)
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        #self.Q[state][action] += 1
        next_action = self.select_action(next_state)
        Qsa = self.Q[state][action]
        Qsa_next = self.Q[next_state][next_action]
        self.Q[state][action] = self.update_Q(Qsa, Qsa_next, reward, self.alpha, self.gamma)
        
    