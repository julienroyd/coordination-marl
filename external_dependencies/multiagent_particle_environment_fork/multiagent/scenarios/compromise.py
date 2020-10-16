import numpy as np
from multiagent.core import World, Agent, Landmark, ConnectionLine
from multiagent.scenario import BaseScenario
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Scenario(BaseScenario):
    def make_world(self, n_agents=2, line_length=0.3, show_all_landmarks=True, use_dense_rewards=False):
        world = World()
        world.use_dense_rewards = use_dense_rewards
        world.show_all_landmarks = show_all_landmarks

        # set any world properties first
        num_agents = n_agents
        num_landmarks = n_agents
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.clip_positions = np.array([[-world.scale, -world.scale], [world.scale, world.scale]])
            agent.collide = False
        # Add a line between the agents
        world.lines = []
        for agent_i in world.agents:
            for agent_j in world.agents:
                if agent_i is agent_j:
                    continue
                elif any([(agent_i in line.entities) and (agent_j in line.entities) for line in world.lines]):
                    continue
                else:
                    world.lines.append(ConnectionLine(agent_i, agent_j, max_length=line_length, hard=False, elasticity=5.))
                    world.lines.append(ConnectionLine(agent_i, agent_j, max_length=line_length, hard=False, elasticity=5., width=3.))
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size *= 2
            landmark.has_been_reached = False
        # make initial conditions
        self.reset_world(world)

        return world

    def reset_world(self, world):
        self.t = 0

        cm = plt.cm.get_cmap('viridis')

        # agent colors
        for i, agent in enumerate(world.agents):
            color = np.array([0.721, 0.462, 0.196]) if i == 0 else np.array([0.937, 0.701, 0.462])
            agent.color = color
        # landmark colors
        for i, landmark in enumerate(world.landmarks):
            color = np.array([0.2, 0.2, 0.2]) if i == 0 else np.array([0.6, 0.6, 0.6])
            landmark.color = color

        # set random initial positions for agents
        init_order = list(range(len(world.agents)))
        np.random.shuffle(init_order)

        for i in init_order:
            world.agents[i].state.p_vel = np.zeros(world.dim_p)
            world.agents[i].state.c = np.zeros(world.dim_c)

            legit_init = False
            while not legit_init:
                world.agents[i].state.p_pos = np.random.uniform(-world.scale, world.scale, world.dim_p)
                legit_init = not any([line.exceeds_max_length() for line in world.lines])

        # set random initial positions for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.has_been_reached = False
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.state.p_pos = np.random.uniform(-world.scale, world.scale, world.dim_p)

        for line in world.lines:
            line.save_entities_colors()
            line.movable = True

    def post_step(self, world):
        self.t += 1
        # sets the activated landmarks green and unactivated landmarks gray
        for i, landmark in enumerate(world.landmarks):
            if landmark.has_been_reached:
                landmark.has_been_reached = False
                landmark.state.p_pos = np.random.uniform(-world.scale, world.scale, world.dim_p)

    def benchmark_data(self, agent, world):
        raise NotImplemented

    def dense_reward(self, agent, world):
        # Agents are penalized based on the distance with their landmark
        rew = 0
        agent_i = int(agent.name.strip('agent '))
        dist = np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[agent_i].state.p_pos)))
        rew -= dist * 0.1
        return rew

    def sparse_reward(self, agent, world):
        # Agents are rewarded for being inside their landmark
        rew = 0

        agent_i = int(agent.name.strip('agent '))
        dist = np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[agent_i].state.p_pos)))
        agent_in_landmark = dist < world.landmarks[agent_i].size
        rew += 10 * int(agent_in_landmark)

        if agent_in_landmark:
            world.landmarks[agent_i].has_been_reached = True

        return rew

    def reward(self, agent, world):
        rew = self.sparse_reward(agent, world)
        if world.use_dense_rewards:
            rew += self.dense_reward(agent, world)

        return rew

    def observation(self, agent, world):
        agent_i = int(agent.name.strip('agent '))
        def rotate_list(l, n):
            return l[-n:] + l[:-n]

        if world.show_all_landmarks:
            # get all landmarks positions
            landmark_pos = [landmark.state.p_pos - agent.state.p_pos for landmark in world.landmarks]
            landmark_pos = rotate_list(landmark_pos, agent_i)
        else:
            # get position of its landmark
            landmark_pos = [world.landmarks[agent_i].state.p_pos - agent.state.p_pos]

        # communication and position of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_pos = rotate_list(other_pos, agent_i)

        x = [agent.state.p_vel] + [agent.state.p_pos] + landmark_pos + other_pos + comm
        return np.concatenate(x)
