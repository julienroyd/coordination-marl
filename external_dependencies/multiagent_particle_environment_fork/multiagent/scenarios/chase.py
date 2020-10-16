# CODE TAKEN FROM multiagent.scenarios.simple_tag.py
import numpy as np
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario
from multiagent.policy import RunnerPolicy


class Scenario(BaseScenario):
    def make_world(self, n_preds=2, n_preys=1, prey_variance=0., use_dense_rewards=False):
        world = World()
        world.use_dense_rewards = use_dense_rewards

        # set any world properties first
        num_agents = n_preds + n_preys
        num_landmarks = 0  # No landmark to avoid lucky catches since the prey is scripted and cannot avoid them
        # add policy for always_scripted agents
        self.runner_policy = RunnerPolicy(var=prey_variance)
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.adversary = False if i < n_preds else True
            agent.size = 0.04 if agent.adversary else 0.05
            agent.accel = 1.5 if agent.adversary else 1.
            agent.max_speed = 1.5 if agent.adversary else 1.
            agent.always_scripted = True if agent.adversary else False
            agent.clip_positions = np.array([[-world.scale, -world.scale], [world.scale, world.scale]])
            if agent.adversary and agent.always_scripted:
                agent.action_callback = self.runner_policy.action
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size *= 2

        # make initial conditions
        self.reset_world(world)

        return world


    def reset_world(self, world):
        self.t = 0
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.301, 0.780, 0.670]) if agent.adversary else np.array([0.8, 0.5, 0.2])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        while True:
            # set random initial states
            for agent in world.agents:
                if not(agent.adversary):
                    agent.state.p_pos = np.random.uniform(-0.8 * world.scale, +0.8 * world.scale, world.dim_p)
                else:
                    agent.state.p_pos = np.random.uniform(-world.scale, +world.scale, world.dim_p)
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
            for i, landmark in enumerate(world.landmarks):
                landmark.state.p_pos = np.random.uniform(-0.9 * world.scale, +0.9 * world.scale, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
            # checks for overlaps between agent's initial positions
            overlap = 0
            for agent_i in world.agents:
                if not agent_i.adversary:  # agent_i is an prey
                    continue
                for agent_j in world.agents:
                    if agent_j.adversary:  # agent_j is a predator
                        continue
                    if agent_i is agent_j:
                        continue
                    if np.sqrt(np.sum(np.square(agent_i.state.p_pos - agent_j.state.p_pos))) < 1.5*(agent_i.size + agent_j.size):
                        overlap += 1
            # if there is a single overlap, we re-do the position initialization
            if overlap > 0:
                continue
            else:
                break

    def post_step(self, world):
        self.t += 1

    def benchmark_data(self, agent, world):
        raise NotImplemented

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def agent_dense_reward(self, agent, world):
        rew = 0
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        for ag in agents:
            rew -= 0.1 * min([np.sqrt(np.sum(np.square(ag.state.p_pos - adv.state.p_pos))) for adv in adversaries])

        return rew

    def agent_sparse_reward(self, agent, world):
        rew = 0
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10

        return rew

    def reward(self, agent, world):
        rew = self.agent_sparse_reward(agent, world)
        if world.use_dense_rewards:
            rew += self.agent_dense_reward(agent, world)

        if agent.adversary:
            assert agent.always_scripted
            rew = 0.

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication and position of all other agents
        comm = []
        other_pos = []
        prey_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if other.adversary:
                prey_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + prey_vel + comm)
