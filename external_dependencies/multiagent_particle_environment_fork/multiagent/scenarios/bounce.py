import numpy as np
from multiagent.core import World, Agent, Landmark, ConnectionLine
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, episode_length=100, line_length=0.5, use_dense_rewards=False):
        world = World()
        world.use_dense_rewards = False  # There are no dense reward in this task

        # set any world properties first
        num_landmarks = 2
        num_agents = 2
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.clip_positions = np.array([[-world.scale, -world.scale], [world.scale, 0.5 * world.scale]])
            agent.collide = False
        # Add a line between the agents
        world.lines = [ConnectionLine(world.agents[0], world.agents[1], max_length=line_length, hard=True, elasticity=2., width=3.)]
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            if i == 0:
                landmark.role = 'bouncing-ball'
                landmark.collide = True
                landmark.movable = True
                landmark.size *= 0.75
                landmark.accel = 5.
                landmark.max_speed = 0.5
                landmark.damping = 0.
                landmark.has_bounced = False
            elif i == 1:
                landmark.role = 'target'
                landmark.collide = False
                landmark.movable = False
                landmark.size *= 4
            else:
                raise ValueError('There should be only two landmarks')

        # make initial conditions
        self.reset_world(world)
        self.episode_length = episode_length
        self.ball_has_bounced = False

        return world

    def reset_world(self, world):
        self.t = 0
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.8, 0.5, 0.2])  # violet
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            if landmark.role == 'bouncing-ball':
                landmark.color = np.array([0.8, 0.0, 0.0])*0  # black
            elif landmark.role == 'target':
                landmark.color = np.array([1., 0.9, 0.6])  # beige

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

        # set random initial states for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.array([np.random.uniform(-world.scale, world.scale), 0.85])
            landmark.state.p_vel = np.zeros(world.dim_p)
            if landmark.role == 'bouncing-ball':
                landmark.has_bounced = False

        for line in world.lines:
            line.save_entities_colors()
            line.movable = True
            line.hard = True

        self.ball_has_bounced = False

    def post_step(self, world):
        self.t += 1
        # at the halftime, drop the ball
        if self.t == int(0.5 * self.episode_length):
            for landmark in world.landmarks:
                if landmark.role == 'bouncing-ball':
                    landmark.state.p_vel = landmark.max_speed * np.array([0., -1.])
            # world.lines[0].movable = False

    def benchmark_data(self, agent, world):
        raise NotImplemented

    def check_if_done(self, world):
        return self.ball_has_bounced

    def ball_target_will_intersect(self, ball, target):
        # see https://math.stackexchange.com/questions/311921/get-location-of-vector-circle-intersection

        # ray (ball) params
        x_0 = ball.state.p_pos[0]
        y_0 = ball.state.p_pos[1]
        v_x = ball.state.p_vel[0]
        v_y = ball.state.p_vel[1]

        # circle (target) params
        h = target.state.p_pos[0]
        k = target.state.p_pos[1]
        r = target.size

        # terms to solve the quadratic
        a = (v_x ** 2) + (v_y ** 2) + 1e-8
        b = (2. * v_x * (x_0 - h)) + (2. * v_y * (y_0 - k))
        c = (x_0 - h)**2 + (y_0 - k)**2 - r**2

        discriminant = b**2 - 4. * a * c

        if discriminant < 0:
            # no solution
            return False

        else:
            t1 = (- b + np.sqrt(discriminant)) / (2. * a)
            t2 = (- b + np.sqrt(discriminant)) / (2. * a)

            if t1 > 0. and t2 > 0.:
                return True
            else:
                return False

    def rew_ball_in_target(self, world):
        ball = world.landmarks[0]
        target = world.landmarks[1]

        if ball.has_bounced:
            self.ball_has_bounced = True
            if self.ball_target_will_intersect(ball, target):

                rew = 10.
                target.color = np.array([1,1,0]) #np.array([0.5, 0.8, 0.3])
                return rew

        rew = 0.
        target.color = np.array([1., 0.9, 0.5])
        return rew

    def rew_bounce(self, world):
        ball = world.landmarks[0]
        rew = 0.

        if ball.has_bounced:
            self.ball_has_bounced = True
            rew = 0.1

            # ray (ball) params
            x_0 = ball.state.p_pos[0]
            y_0 = ball.state.p_pos[1]
            v_x = ball.state.p_vel[0]
            v_y = ball.state.p_vel[1]

            # solutions for intersection between ray and world limits
            t_down = (-1. - y_0) / (v_y + 1e-5)
            t_left = (-1. - x_0) / (v_x + 1e-5)
            t_right = (1. - x_0) / (v_x + 1e-5)
            t_up = (1. - y_0) / (v_y + 1e-5)

            # ball is heading towards the top
            if t_up > 0. and (-1 < v_x * t_up + x_0 and v_x * t_up + x_0 < 1):
                rew += 0.2

            # ball is heading towards the left side
            if t_left > 0. and (-1 < v_y * t_left + y_0 and v_y * t_left + y_0 < 1):
                rew += 0.1

            # ball is heading towards the right side
            if t_right > 0. and (-1 < v_y * t_right + y_0 and v_y * t_right + y_0 < 1):
                rew += 0.1

            # ball is heading towards the bottom
            if t_down > 0. and (-1 < v_x * t_down + x_0 and v_x * t_down + x_0 < 1):
                rew += 0.

        return rew

    def reward(self, agent, world):
        # Agents are rewarded when the ball reaches the target
        rew = 0
        rew += self.rew_ball_in_target(world)
        rew += self.rew_bounce(world)
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # position of all other agents
        other_pos = []
        for other_agent in world.agents:
            if other_agent is agent:
                continue
            else:
                other_pos.append(other_agent.state.p_pos - agent.state.p_pos)
        # relative time of the episode
        time = np.array([self.t / self.episode_length]).reshape(1,1)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + time)
