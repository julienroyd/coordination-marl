import numpy as np
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario
import matplotlib.pyplot as plt


class Scenario(BaseScenario):
    def make_world(self, n_agents=3, by_stander=False):
        world = World()
        # set any world properties first
        world.dim_c = 0
        num_agents = n_agents
        num_walls = 4
        # add landmarks
        world.landmarks = []
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        world.goals = ['top', 'bottom', 'left', 'right']
        cm = plt.cm.get_cmap('rainbow')
        world.colors = [np.array(cm(float(i) / float(len(world.agents)))[:3]) for i in range(len(world.agents))]
        for i, agent in enumerate(world.agents):
            if by_stander and i == 0:
                agent.goal = None
            else:
                agent.goal = np.random.choice(world.goals)
            agent.name = 'agent %d' % i
            agent.silent = True
            agent.collide = True
            agent.size = 0.05
            agent.accel = 1.5
            agent.max_speed = 1.5
            agent.adversary = False
            agent.clip_positions = np.array([[-1.02, -1.02], [1.02, 1.02]])
            agent.is_colliding = {other_agent.name: False for other_agent in world.agents if agent is not other_agent}
        # add walls to force the "leader" through a longer path
        world.walls = [Wall() for i in range(num_walls)]
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        init_pos = ([[-1., -0.1 + 1.2 * world.agents[0].size], [1., 0.1 - 1.2 * world.agents[0].size]], # Horizontal street
                    [[-0.1 + 1.2 * world.agents[0].size, -1.], [0.1 - 1.2 * world.agents[0].size, 1.]]  # Vertical street
        )

        # set random initial states
        for i, agent in enumerate(world.agents):
            agent.color = world.colors[i]
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            if agent.goal in world.goals:
                agent.goal = np.random.choice(world.goals)
            if np.random.binomial(n=1, p=0.5):
                agent.state.p_pos = np.random.uniform(init_pos[0][0], init_pos[0][1], (2,))
            else:
                agent.state.p_pos = np.random.uniform(init_pos[1][0], init_pos[1][1], (2,))

        # set walls
        walls_properties = [
            # top-left block
            {'orient': 'V', 'axis_pos': -0.55, 'endpoints': [+0.1, +1.0], 'width': 0.9},
            # top-right block
            {'orient': 'V', 'axis_pos': +0.55, 'endpoints': [+0.1, +1.0], 'width': 0.9},
            # bottom-left block
            {'orient': 'V', 'axis_pos': -0.55, 'endpoints': [-0.1, -1.0], 'width': 0.9},
            # bottom-right block
            {'orient': 'V', 'axis_pos': +0.55, 'endpoints': [-0.1, -1.0], 'width': 0.9},
        ]

        for i, wall in enumerate(world.walls):
            wall.bouncy = False
            wall.hard = True
            wall.width = walls_properties[i]['width']
            wall.orient = walls_properties[i]['orient']
            wall.axis_pos = walls_properties[i]['axis_pos']
            wall.endpoints = walls_properties[i]['endpoints']

    def post_step(self, world):
        pass

    # def benchmark_data(self, agent, world):
    #     pass

    def is_collision(self, agent1, agent2):
        if agent1 is agent2 or not agent1.collide or not agent2.collide:
            return False

        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size

        if dist < dist_min:
            agent1.is_colliding[agent2.name] = True
            agent2.is_colliding[agent1.name] = True
            return True

        else:
            agent1.is_colliding[agent2.name] = False
            agent2.is_colliding[agent1.name] = False
            return False

    def count_collisions(self, agent, world):
        n_collisions = 0
        for a in world.agents:
            if self.is_collision(agent, a):
                n_collisions += 1

        # Sets agent's color based on whether it is colliding or not
        if any(agent.is_colliding.values()):
            agent.color = world.colors[int(agent.name.split(' ')[1])] * 0.5
        else:
            agent.color = world.colors[int(agent.name.split(' ')[1])]

        return n_collisions

    def reward(self, agent, world):
        rew = 0.

        # Distance to goal penalty
        if agent.goal == "top":
            dist = np.sqrt(np.sum(np.square(+1. - agent.state.p_pos[1])))
        elif agent.goal == "bottom":
            dist = np.sqrt(np.sum(np.square(-1. - agent.state.p_pos[1])))
        elif agent.goal == "left":
            dist = np.sqrt(np.sum(np.square(-1. - agent.state.p_pos[0])))
        elif agent.goal == "right":
            dist = np.sqrt(np.sum(np.square(+1. - agent.state.p_pos[0])))
        elif agent.goal is None:
            dist = 0.
        else:
            raise NotImplemented
        rew -= dist

        # Collision penalty
        if agent.collide:
            rew -= 10. * self.count_collisions(agent, world)

        return rew

    def observation(self, agent, world):
        # position of all other agents
        other_pos = []
        for other_agent in world.agents:
            if other_agent is agent:
                continue
            else:
                other_pos.append(other_agent.state.p_pos - agent.state.p_pos)
        goal = []
        if agent.goal is not None:
            goal = np.zeros(shape=(len(world.goals),), dtype=np.float)
            goal[world.goals.index(agent.goal)] = 1.

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + [goal])
