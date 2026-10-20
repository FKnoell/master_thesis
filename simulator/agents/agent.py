from utils import *

# super class of scheduling agnet
class Agent(object):
    def __init__(self, pidle=0.0, pdyn=1.0, carbon_schedule=None):
        self.pidle = pidle
        self.pdyn = pdyn
        self.carbon_schedule = carbon_schedule or {0: 0.5}

    def carbon_intensity_at(self, current_time):
        available_times = [
            time for time in self.carbon_schedule if time <= current_time]
        if not available_times:
            return self.carbon_schedule[min(self.carbon_schedule)]
        return self.carbon_schedule[max(available_times)]

    def get_action(self, obs):
        print('get_action not implemented')
        exit(1)
