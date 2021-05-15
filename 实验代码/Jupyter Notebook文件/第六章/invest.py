import numpy as np

class InvestEnv(object):
    def __init__(self,init_state=10):
        self.init_state = init_state
        self.action_space = 2
        self.state_space = 50
    def reset(self):
        self.state = self.init_state
    def getState(self):
        return self.state
    def put_reward(self, state, next_state):
        if next_state == state:
            return 0
        if next_state >= self.init_state and next_state > state:
            return 1
        if next_state <= self.init_state and next_state < state:
            return -1
        if next_state < self.init_state and next_state > state:
            return 0.5
        if next_state > self.init_state and next_state < state:
            return -0.5
    def step(self, action):
        if action == 0:
            return round(self.state), self.put_reward(self.state,self.state)
        else:
            pros = np.random.rand()
            if pros < 0.5:
                next_state = self.state * 0.9
                r = self.put_reward(self.state, next_state)
                self.state = next_state
                return round(next_state),r
            else:
                next_state = self.state * 1.1
                r = self.put_reward(self.state, next_state)
                self.state = next_state
                return round(next_state),r