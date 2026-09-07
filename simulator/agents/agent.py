from utils import *

# super class of scheduling agnet
class Agent(object):
    def __init__(self, pidle=0.0, pdyn=1.0):
        self.pidle = pidle
        self.pdyn = pdyn

    def get_action(self, obs):
        print('get_action not implemented')
        exit(1)
