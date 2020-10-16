import numpy as np

# defines scenario upon which the world is built
class BaseScenario(object):
    # create elements of the world
    def make_world(self, world_params):
        raise NotImplementedError()
    # create initial conditions of the world
    def reset_world(self, world):
        raise NotImplementedError()
    # used to modify the world after a step
    # (e.g. to keep track of time, change landmark positions, etc.)
    def post_step(self, world):
        return None
    # checks if the objective of the scenario has been achieved
    # (i.e. if the episode should be allowed to continue or should be stopped)
    def check_if_done(self, world):
        return False
