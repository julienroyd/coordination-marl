import numpy as np
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, staged=False, set_trap=True, use_dense_rewards=False):
        world = World()
        world.use_dense_rewards = use_dense_rewards

        # set any world properties first
        self.staged = staged
        self.set_trap = set_trap
        num_agents = 2
        num_landmarks = 2 * num_agents
        self.num_walls = 4 + int(self.staged)

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size *= 4
            landmark.is_activated = False

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            if i == 0:
                agent.role = 'imitator'
                agent.clip_positions = np.array([[-world.scale, .55 * world.scale], [world.scale, world.scale]])
                agent.landmarks = [world.landmarks[2], world.landmarks[3]]
                self.trap_width = agent.size * 4.
            elif i == 1:
                agent.role = 'demonstrator'
                agent.clip_positions = np.array([[-world.scale, -world.scale], [world.scale, .45 * world.scale]])
                agent.landmarks = [world.landmarks[0], world.landmarks[1]]
            else:
                raise ValueError('There should only be two agents in this scenario')

        # add walls to force the "leader" through a longer path
        world.walls = [Wall() for i in range(self.num_walls)]
        if self.set_trap:
            world.trap = Wall()
            world.walls.append(world.trap)

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # colors for agents
        for i, agent in enumerate(world.agents):
            if agent.role == "demonstrator":
                agent.color = np.array([0.1, 0.5, 0.7])
            elif agent.role == "imitator":
                agent.color = np.array([0.4, 0.1, 0.7])
        # set random initial states
        i_activated = np.random.randint(low=0, high=2)
        for agent in world.agents:
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

            if agent.role == 'demonstrator':
                agent.state.p_pos = np.array([0., -0.25 * world.scale])

            elif agent.role == 'imitator':
                agent.state.p_pos = np.array([0., 0.75 * world.scale])

            for j, landmark in enumerate(agent.landmarks):
                landmark.is_activated = False
                landmark.state.p_vel = np.zeros(world.dim_p)
                if j == 0:
                    landmark.state.p_pos = np.array([agent.state.p_pos[0] - 0.725 * world.scale, agent.state.p_pos[1]])
                elif j == 1:
                    landmark.state.p_pos = np.array([agent.state.p_pos[0] + 0.725 * world.scale, agent.state.p_pos[1]])
                if j == i_activated:
                    landmark.is_activated = True
                if landmark.is_activated:
                    landmark.color = np.array([0.5, 0.8, 0.3])
                else:
                    landmark.color = np.array([0.3, 0.3, 0.3])
        # set walls
        walls_properties = [
            # left box
            {'hard': True, 'orient': 'V', 'axis_pos': -0.25 * world.scale, 'endpoints': [-0.60 * world.scale, +0.10 * world.scale], 'width': 0.5 * world.scale - 2 * world.agents[0].size},
            # right box
            {'hard': True, 'orient': 'V', 'axis_pos': +0.25 * world.scale, 'endpoints': [-0.60 * world.scale, +0.10 * world.scale], 'width': 0.5 * world.scale - 2 * world.agents[0].size},
            # top part
            {'hard': True, 'orient': 'H', 'axis_pos': +0.90 * world.scale, 'endpoints': [-world.scale, world.scale], 'width': 0.2 * world.scale},
            # bottom part
            {'hard': True, 'orient': 'H', 'axis_pos': +0.60 * world.scale, 'endpoints': [-world.scale, world.scale], 'width': 0.2 * world.scale},
        ]
        if self.staged:
            walls_properties.append(
                {'hard': True, 'orient': 'V', 'axis_pos': +0., 'endpoints': [-0.60 * world.scale, -0.25 * world.scale - world.agents[0].size] if world.agents[0].landmarks[0].is_activated else [-0.25 * world.scale + world.agents[0].size, +0.10 * world.scale], 'width': 0.1 * world.scale})

        if self.set_trap:
            walls_properties.append(
                {'hard': False, 'orient': 'H', 'axis_pos': 0.75 * world.scale, 'endpoints': [-self.trap_width / 2., self.trap_width / 2.], 'width': 2. * world.agents[0].size}
            )
        self.trap_deployed = False

        for i, wall in enumerate(world.walls):
            self.set_wall_properties(wall, walls_properties[i])

    def set_wall_properties(self, wall, wall_properties):
        wall.bouncy = False
        wall.hard = wall_properties['hard']
        wall.width = wall_properties['width']
        wall.orient = wall_properties['orient']
        wall.axis_pos = wall_properties['axis_pos']
        wall.endpoints = wall_properties['endpoints']

    def post_step(self, world):
        # sets the activated landmarks green and unactivated landmarks gray
        for landmark in world.landmarks:
            if landmark.is_activated:
                landmark.color = np.array([0.5, 0.8, 0.3])  # green
            else:
                landmark.color = np.array([0.3, 0.3, 0.3])  # gray

        if not self.trap_deployed and self.trap_should_be_deployed(world):
            self.deploy_trap(world)

    def trap_should_be_deployed(self, world):
        # the trap should be deployed if the 'imitator' has committed itself to one side or another
        for agent in world.agents:
            if agent.role == "imitator" and abs(agent.state.p_pos[0]) > (self.trap_width / 2.) + agent.size and self.set_trap:
                    return True

        return False

    def deploy_trap(self, world):
        assert self.set_trap and not self.trap_deployed
        world.trap.hard = True
        self.trap_deployed = True

    def benchmark_data(self, agent, world):
        raise NotImplemented

    def dense_reward(self, agent):
        rew = 0.

        for landmark in agent.landmarks:
            if landmark.is_activated:
                # agent is proportionally penalised for being distant (only in x-dimension) from the landmark
                rew -= 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos[0] - landmark.state.p_pos[0])))

        return rew

    def sparse_reward(self, agent):
        rew = 0.

        for landmark in agent.landmarks:
            if landmark.is_activated:
                # agent is rewarded for being in the landmark
                if np.sum(np.square(agent.state.p_pos - landmark.state.p_pos)) < landmark.size**2:
                    rew += 0.1

        return rew

    def reward(self, agent, world):
        rew = self.sparse_reward(agent)
        if world.use_dense_rewards:
            rew += self.dense_reward(agent)
        return rew

    def observation(self, agent, world):
        # get positions of landmarks that concern the current agent
        landmarks_pos = []
        activated_landmarks = []
        for landmark in agent.landmarks:
            landmarks_pos.append(landmark.state.p_pos - agent.state.p_pos)
            if agent.role == 'demonstrator':
                activated_landmarks.append(np.array([landmark.is_activated], dtype=np.float))
        # position of all other agents
        agents_pos = []
        for other_agent in world.agents:
            if other_agent is agent:
                continue
            elif agent.role == 'imitator':
                agents_pos.append(other_agent.state.p_pos) # we give the absolute position of the other agent
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + landmarks_pos + activated_landmarks + agents_pos)
