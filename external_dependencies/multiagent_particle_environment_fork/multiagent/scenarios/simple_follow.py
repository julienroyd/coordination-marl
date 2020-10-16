import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, use_dense_rewards=True):
        world = World()
        world.use_dense_rewards = True  # We always use_dense_rewards in this task

        # set any world properties first
        num_landmarks = 3
        num_agents = 2
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.clip_positions = np.array([[-world.scale, -world.scale], [world.scale, world.scale]])
            agent.collide = False
            if i == 0:
                agent.role = "runner"
            elif i == 1:
                agent.role = "pursuer"
            else:
                raise ValueError('There should only be two agents in this scenario')
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size *= 4
            landmark.is_activated = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            if agent.role == "runner":
                agent.color = np.array([0.1, 0.5, 0.7])
            elif agent.role == "pursuer":
                agent.color = np.array([0.4, 0.1, 0.7])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-world.scale, world.scale, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # properties for agents
        theta = 2 * np.pi / len(world.landmarks)
        dist = 0.5 * world.scale
        i_activated = np.random.randint(low=0, high=len(world.landmarks))
        for i, landmark in enumerate(world.landmarks):
            landmark.is_activated = True if i == i_activated else False
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.state.p_pos = np.array([dist * np.cos(i * theta), dist * np.sin(i * theta)])
            if landmark.is_activated:
                landmark.color = np.array([0.5, 0.8, 0.3])  # green
            else:
                landmark.color = np.array([0.3, 0.3, 0.3])  # gray

    def post_step(self, world):
        # sets the activated landmarks green and unactivated landmarks gray
        for landmark in world.landmarks:
            if landmark.is_activated:
                landmark.color = np.array([0.5, 0.8, 0.3])  # green
            else:
                landmark.color = np.array([0.3, 0.3, 0.3])  # gray

    def benchmark_data(self, agent, world):
        raise NotImplemented

    def runner_reward(self, agent, world):
        rew = 0.
        for i, landmark in enumerate(world.landmarks):

            if  landmark.is_activated:
                # dense reward leading the agent towards activated landmark
                rew -= 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos)))

                # If the landmark is activated and the agent is inside it, we give it a reward of 1.
                if np.sum(np.square(agent.state.p_pos - landmark.state.p_pos)) < landmark.size**2:

                    rew += 1.
                    while True:
                        j_activated = np.random.randint(low=0, high=len(world.landmarks))
                        if i != j_activated: break
                    for j, landmark in enumerate(world.landmarks):
                        landmark.is_activated = True if j == j_activated else False

        return rew

    def pursuer_reward(self, agent, world):
        # Adversaries are rewarded for being closed to agents (for following them)
        rew = 0.
        runners = [a for a in world.agents if a.role == "runner"]
        rew -= 0.1 * min([np.sqrt(np.sum(np.square(agent.state.p_pos - runner.state.p_pos))) for runner in runners])
        return rew

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.runner_reward(agent, world) if agent.role == "runner" else self.pursuer_reward(agent, world)
        return main_reward

    def observation(self, agent, world):
        # position of all landmarks and activation_state
        landmarks_pos = []
        activated_landmarks = []
        if agent.role == "runner":
            for landmark in world.landmarks:
                landmarks_pos.append(landmark.state.p_pos - agent.state.p_pos)
                activated_landmarks.append(np.array([landmark.is_activated], dtype=np.float))
        # position of all other agents
        agents_pos = []
        if agent.role == "pursuer":
            for other_agent in world.agents:
                if other_agent is agent:
                    continue
                else:
                    agents_pos.append(other_agent.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + landmarks_pos + activated_landmarks + agents_pos)
