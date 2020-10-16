from queue import Queue
import numpy as np
from numpy import sin, cos
import scipy.integrate as integrate
from pyglet.window import key
from multiagent.core import Action

# individual agent policy
class Policy(object):
    def __init__(self):
        pass
    def action(self, *args):
        raise NotImplementedError()

# interactive policy based on keyboard input
# hard-coded to deal only with movement, not communication
class InteractivePolicy(Policy):
    def __init__(self, env, viewer_id):
        super(InteractivePolicy, self).__init__()
        self.env = env
        # hard-coded keyboard events
        self.move = [False for i in range(4)]
        self.comm = [False for i in range(env.world.dim_c)]
        # register keyboard events with this environment's window
        env.viewers[viewer_id].window.on_key_press = self.key_press
        env.viewers[viewer_id].window.on_key_release = self.key_release

    def action(self, *args):
        """
        Arguments are ignored for InteractivePolicy. However, the call to action_callback(agent, self)
        in multiagent.core.py requires that this callback method accepts an agent and world if provided.
        """
        action = Action()
        # ignore observation and just act based on keyboard events
        if self.env.discrete_action_input:
            u = 0
            if self.move[0]: u = 1
            if self.move[1]: u = 2
            if self.move[2]: u = 4
            if self.move[3]: u = 3
        else:
            u = np.zeros(2)
            if self.move[0]: u[0] += 1.
            if self.move[1]: u[0] -= 1.
            if self.move[3]: u[1] += 1.
            if self.move[2]: u[1] -= 1.
        action.u = u
        return action

    # keyboard event callbacks
    def key_press(self, k, mod):
        if k==key.RIGHT:  self.move[0] = True
        if k==key.LEFT: self.move[1] = True
        if k==key.DOWN:    self.move[2] = True
        if k==key.UP:  self.move[3] = True
    def key_release(self, k, mod):
        if k==key.RIGHT:  self.move[0] = False
        if k==key.LEFT: self.move[1] = False
        if k==key.DOWN:    self.move[2] = False
        if k==key.UP:  self.move[3] = False


class RunnerPolicy(Policy):
    """
    Policy for prey in simple_tag setups
    Simply runs away from the adversaries and the limits of the environment.
    Driven by repulsive forces inversely proportional to its distance with those entities.
    Only creates movement action, not communication.
    """
    def __init__(self, max_force=1., var=0.):
        self.max_force = max_force
        self.var = var
        super(RunnerPolicy, self).__init__()

    def action(self, agent, world):
        action = Action()
        forces = []
        epsilon = 1e-5  # for numerical stability

        # Distances from environment borders
        d_right = agent.state.p_pos[0] - (1. + epsilon)
        d_left = agent.state.p_pos[0] + (1. + epsilon)
        d_up = agent.state.p_pos[1] - (1. + epsilon)
        d_down = agent.state.p_pos[1] + (1. + epsilon)

        # Forces from predator agents
        for other_agent in world.agents:
            if not other_agent.adversary and agent is not other_agent:
                force_vec = (agent.state.p_pos - other_agent.state.p_pos).reshape((1,2))
                force_norm = np.sqrt(np.sum(np.square(force_vec)))
                force_agent = (force_vec * (1. + epsilon)) / (force_norm + epsilon) ** 3
                forces.append(force_agent)

        # Forces from environment borders
        forces.append(np.array([np.sign(d_right) / (d_right + epsilon) ** 2, 0.]).reshape(1,2))
        forces.append(np.array([np.sign(d_left) / (d_left + epsilon) ** 2, 0.]).reshape(1,2))
        forces.append(np.array([0., np.sign(d_up) / (d_up + epsilon) ** 2]).reshape(1,2))
        forces.append(np.array([0., np.sign(d_down) / (d_down + epsilon) ** 2]).reshape(1,2))

        # Accumulate all forces to get final force (and makes sure it is not bigger than max force)
        final_force = np.sum(np.vstack(forces), axis=0)
        final_force = final_force if np.sqrt(np.sum(final_force ** 2)) < self.max_force \
            else self.max_force * (final_force / np.sqrt(np.sum(final_force ** 2)))

        # Disturb final force by adding gaussian noise
        final_force += np.random.multivariate_normal([0., 0.], [[self.var, 0.], [0., self.var]])

        action.u = final_force

        return action


class RusherPolicy(Policy):
    """
    Policy for predators in simple_tag setups
    Simply rushes towards the prey(s).
    Driven by attractive forces proportional to its distance with the prey(s).
    Only creates movement action, not communication.
    """
    def __init__(self, max_force=1.):
        self.max_force = max_force
        super(RusherPolicy, self).__init__()

    def action(self, agent, world):
        action = Action()
        forces = []
        epsilon = 1e-5  # for numerical stability

        # Forces from prey agents
        for other_agent in world.agents:
            if other_agent.adversary and agent is not other_agent:
                force_vec = (other_agent.state.p_pos - agent.state.p_pos).reshape((1,2))
                force_norm = np.sqrt(np.sum(np.square(force_vec)))
                force_agent = (force_vec * (1. + epsilon)) / (force_norm + epsilon) ** 3
                forces.append(force_agent)

        # Accumulate all forces to get final force (and makes sure it is not bigger than max force)
        final_force = np.sum(np.vstack(forces), axis=0)
        final_force = final_force if np.sqrt(np.sum(final_force ** 2)) < self.max_force \
            else self.max_force * (final_force / np.sqrt(np.sum(final_force ** 2)))

        action.u = final_force

        return action


class DoublePendulumPolicy(Policy):
    """
    Policy for any particle
    Follows a trajectory computed by double-pendulum dynamics
    First it computes the trajectory of the particle. It then transforms this trajectory in shifts.
    The trajectory shift are given as action forces to the agent (so DOES NOT perfectly follow the pendulum trajectory)
    Restricted to movement action: no communication.
    """
    def __init__(self, length1=1., length2=1., mass1=1., mass2=1., gravity=3.,
                 init_th1=120., init_th2=-10., init_w1=100., init_w2=-50., time_step=0.05, time_end=100):
        super(DoublePendulumPolicy, self).__init__()

        # Physical properties
        self.length1 = length1
        self.length2 = length2
        self.mass1 = mass1
        self.mass2 = mass2
        self.gravity = gravity

        # Initial conditions
        self.init_th1 = init_th1
        self.init_th2 = init_th2
        self.init_w1 = init_w1
        self.init_w2 = init_w2

        # Simulation length and resolution
        self.time_step = time_step
        self.time_end = time_end

        self.precompute_actions()

    def precompute_actions(self):
        # Runs a simulation and keep it in memory
        # The trajectory given by the simulation will be followed step by step
        vx1, vx2, vy1, vy2 = self.simulate()
        self.vshifts = np.vstack(self.convert_trajectory_to_forces(vx2, vy2)).T
        self.precomputed_actions = Queue()
        for vshift in self.vshifts:
            self.precomputed_actions.put(vshift)

    def simulate(self):
        # create a time array from 0..100 sampled at 0.05 second steps
        t = np.arange(0.0, self.time_end, self.time_step)

        # initial state
        init_state = np.radians([self.init_th1, self.init_th1, self.init_w1, self.init_w2])

        def derivs(state, t):
            dydx = np.zeros_like(state)
            dydx[0] = state[1]

            del_ = state[2] - state[0]
            den1 = (self.mass1 + self.mass2) * self.length1 - self.mass2 * self.length1 * cos(del_) * cos(del_)
            dydx[1] = (self.mass2 * self.length1 * state[1] * state[1] * sin(del_) * cos(del_) +
                       self.mass2 * self.gravity * sin(state[2]) * cos(del_) +
                       self.mass2 * self.length2 * state[3] * state[3] * sin(del_) -
                       (self.mass1 + self.mass2) * self.gravity * sin(state[0])) / den1

            dydx[2] = state[3]

            den2 = (self.length2 / self.length1) * den1
            dydx[3] = (-self.mass2 * self.length2 * state[3] * state[3] * sin(del_) * cos(del_) +
                       (self.mass1 + self.mass2) * self.gravity * sin(state[0]) * cos(del_) -
                       (self.mass1 + self.mass2) * self.length1 * state[1] * state[1] * sin(del_) -
                       (self.mass1 + self.mass2) * self.gravity * sin(state[2])) / den2

            return dydx

        # integrate the ODE using scipy.integrate.
        y = integrate.odeint(derivs, init_state, t)

        # converts the angular velocities to cartesian velocities
        vx1 = self.length1 * sin(y[:, 1])
        vy1 = -self.length1 * cos(y[:, 1])

        vx2 = self.length2 * sin(y[:, 3]) + vx1
        vy2 = -self.length2 * cos(y[:, 3]) + vy1

        return vx1, vx2, vy1, vy2

    def convert_trajectory_to_forces(self, x, y):
        x_shifts = [x[i + 1] - x[i] for i in range(len(x) - 1)]
        y_shifts = [y[i + 1] - y[i] for i in range(len(y) - 1)]

        return x_shifts, y_shifts

    def action(self, agent, world):
        action = Action()
        action.u = 100 * self.precomputed_actions.get()
        return action
