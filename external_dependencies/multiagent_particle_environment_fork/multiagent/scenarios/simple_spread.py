import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self, n_agents=3, use_dense_rewards=False, shuffle_landmarks=False, color_objects=False,
                   small_agents=False):
        world = World()
        world.use_dense_rewards = use_dense_rewards
        self.shuffle_landmarks = shuffle_landmarks
        self.color_objects = color_objects
        self.small_agents = small_agents

        # set any world properties first
        num_agents = n_agents
        num_landmarks = n_agents
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.clip_positions = np.array([[-world.scale, -world.scale], [world.scale, world.scale]])
            agent.is_colliding = {other_agent.name:False for other_agent in world.agents if agent is not other_agent}
            if not self.small_agents:
                agent.size *= 3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            if not self.small_agents:
                landmark.size *= 0.5
            else:
                landmark.size *= 2

        # make initial conditions
        self.reset_world(world)

        return world

    def reset_world(self, world):
        colors = [np.array([0.8, 0., 0.]), np.array([0., 0.8, 0.]), np.array([0., 0., 0.8])]
        # random properties for agents
        for i, agent in enumerate(world.agents):
            if not self.color_objects:
                agent.color = np.array([0.8, 0.5, 0.2])
                agent._color = agent.color
            else:
                agent.color = colors[i]
                agent._color = agent.color
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            if not self.color_objects:
                landmark.color = np.array([0.75, 0.75, 0.75])
            else:
                landmark.color = colors[i]
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-world.scale, +world.scale, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            if self.shuffle_landmarks:
                agent.point_of_vue = np.random.permutation(len(world.landmarks))
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-world.scale, +world.scale, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        raise NotImplemented

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
        for a_i in world.agents:
            for a_j in world.agents:
                if self.is_collision(a_i, a_j):
                    n_collisions += 0.5

        # Sets agent's color based on whether it is colliding or not
        if any(agent.is_colliding.values()):
            agent.color = np.array([0.2, 0.2, 0.2])
        else:
            agent.color = agent._color

        return n_collisions

    def dense_reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists) * 0.1
        return rew

    def sparse_reward(self, agent, world):
        rew = 0
        if agent.collide:
            rew -= 1. * self.count_collisions(agent, world)

        for l in world.landmarks:
            if not self.small_agents:
                agents_in = [np.sum(np.square(a.state.p_pos - l.state.p_pos)) < a.size ** 2 for a in world.agents]
            else:
                agents_in = [np.sum(np.square(a.state.p_pos - l.state.p_pos)) < l.size**2 for a in world.agents]
            rew += 1. if any(agents_in) else 0.

        return rew

    def reward(self, agent, world):
        rew = self.sparse_reward(agent, world)
        if world.use_dense_rewards:
            rew += self.dense_reward(agent, world)
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication and position of all other agentsof all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        if self.shuffle_landmarks:
            entity_pos = np.array(entity_pos)[agent.point_of_vue]
            entity_pos = list(entity_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
