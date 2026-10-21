from utils import *

# super class of scheduling agnet
class Agent(object):
    def __init__(self, pidle=0.0, pdyn=1.0, carbon_schedule=None):
        self.pidle = pidle
        self.pdyn = pdyn
        self.carbon_schedule = carbon_schedule if carbon_schedule is not None else {0: 0.0}

    def carbon_intensity_at(self, current_time):
        bucket_start = current_time - (current_time % 60000)
        return self.carbon_schedule.get(
            bucket_start, self.carbon_schedule[min(self.carbon_schedule)])

    def get_action(self, obs):
        print('get_action not implemented')
        exit(1)
