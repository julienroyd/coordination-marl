import numpy as np
import seaborn as sns

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

class Wall(object):
    def __init__(self, orient='H', axis_pos=0.0, endpoints=(-1, 1), width=0.1, hard=True):
        # orientation: 'H'orizontal or 'V'ertical
        self.orient = orient
        # position along axis which wall lays on (y-axis for H, x-axis for V)
        self.axis_pos = axis_pos
        # endpoints of wall (x-coords for H, y-coords for V)
        self.endpoints = np.array(endpoints)
        # width of wall
        self.width = width
        # whether wall is impassable
        self.hard = hard
        # whether a collision with a wall makes you bounce back (pushed by a force), or acts is totally unpenetrable (pushed by clipping)
        self.bouncy = False

    @property
    def hard(self):
        return self.__hard

    @hard.setter
    def hard(self, hard):
        self.__hard = hard
        self.color = np.array([0.0, 0.0, 0.0, 1.0]) if self.hard else np.array([0.0, 0.0, 0.0, 0.4])


class ConnectionLine(object):
    def __init__(self, entity1, entity2, max_length=np.inf, hard=False, movable=True, elasticity=0., width=1.):
        # endpoints of the line
        self.entity1 = entity1
        self.entity2 = entity2
        # the previous endpoints
        self.previous_endpoint1 = np.zeros((2,))
        self.previous_endpoint2 = np.zeros((2,))
        # the original colors of endpoint entities
        self.entity1_color = None
        self.entity2_color = None
        # color of the line
        self.color = np.array([0., 0., 0.])
        # whether line is impassable
        self.hard = hard
        # whether the line can move or not (if not it prevents the entities to move as well)
        self.movable = movable
        # elasticity coefficient (how strongly should it pull the agents towards each other when max_length is exceeded)
        self.elasticity = elasticity
        # width of the line (for rendering only, no physical implications)
        self.width = width
        # the maximum length of the line
        self.max_length = max_length

    @property
    def entities(self):
        return [self.entity1, self.entity2]

    @property
    def start(self):
        return self.entity1.state.p_pos

    @property
    def end(self):
        return self.entity2.state.p_pos

    @property
    def normal(self):
        vector = self.start - self.end
        n1 = np.array([vector[1], -vector[0]]) / np.linalg.norm(vector)
        n2 = np.array([-vector[1], vector[0]]) / np.linalg.norm(vector)
        # returns the normal vector that points upward
        return n1 if n1[1] >= 0.else n2

    def save_entities_colors(self):
        self.entity1_color = np.copy(self.entity1.color)
        self.entity2_color = np.copy(self.entity2.color)

    def save_previous_endpoints(self):
        self.previous_endpoint1 = np.copy(self.entity1.state.p_pos)
        self.previous_endpoint2 = np.copy(self.entity2.state.p_pos)

    def exceeds_max_length(self):
        if self.start is None or self.end is None:
            return False
        else:
            return True if np.linalg.norm(self.start - self.end) > self.max_length else False

    def is_movable(self):
        if self.movable:
            if self.elasticity != 0.:
                return True

            elif not(self.exceeds_max_length()):
                self.entity1.color = np.copy(self.entity1_color)
                self.entity2.color = np.copy(self.entity2_color)
                return True

            elif self.exceeds_max_length():
                self.entity1.color = np.array([.75, .75, .75])
                self.entity2.color = np.array([.75, .75, .75])
                return False

        else:
            self.entity1.color = np.array([0., 0., 0.])
            self.entity2.color = np.array([0., 0., 0.])
            return False

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # index among all entities (important to set for distance caching)
        self.i = 0
        # name 
        self.name = ''
        # properties:
        self.size = 0.05
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # entity can pass through non-hard walls
        self.ghost = False
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = 1.
        self.accel = 1.
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0
        # physical damping
        self.damping = 0.1
        # the x and y limits in which the entity is constrained
        self.clip_positions = None


    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = True
        # cannot observe the world
        self.blind = False
        # flag to distinguish between two teams of agents
        self.adversary = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        self.always_scripted = False

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.walls = []
        self.lines = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # size of half the side of the camera view
        self.scale = 1.
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        # cache distances between all agents (not calculated by default)
        self.cache_dists = False
        self.cached_dist_vect = None
        self.cached_dist_mag = None

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def calculate_distances(self):
        if self.cached_dist_vect is None:
            # initialize distance data structure
            self.cached_dist_vect = np.zeros((len(self.entities),
                                              len(self.entities),
                                              self.dim_p))
            # calculate minimum distance for a collision between all entities
            self.min_dists = np.zeros((len(self.entities), len(self.entities)))
            for ia, entity_a in enumerate(self.entities):
                for ib in range(ia + 1, len(self.entities)):
                    entity_b = self.entities[ib]
                    min_dist = entity_a.size + entity_b.size
                    self.min_dists[ia, ib] = min_dist
                    self.min_dists[ib, ia] = min_dist

        for ia, entity_a in enumerate(self.entities):
            for ib in range(ia + 1, len(self.entities)):
                entity_b = self.entities[ib]
                delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
                self.cached_dist_vect[ia, ib, :] = delta_pos
                self.cached_dist_vect[ib, ia, :] = -delta_pos

        self.cached_dist_mag = np.linalg.norm(self.cached_dist_vect, axis=2)
        self.cached_collisions = (self.cached_dist_mag <= self.min_dists)

    def assign_agent_colors(self):
        n_dummies = 0
        if hasattr(self.agents[0], 'dummy'):
            n_dummies = len([a for a in self.agents if a.dummy])
        n_adversaries = 0
        if hasattr(self.agents[0], 'adversary'):
            n_adversaries = len([a for a in self.agents if a.adversary])
        n_good_agents = len(self.agents) - n_adversaries - n_dummies
        dummy_colors = [(0, 0, 0)] * n_dummies
        adv_colors = sns.color_palette("OrRd_d", n_adversaries)
        good_colors = sns.color_palette("GnBu_d", n_good_agents)
        colors = dummy_colors + adv_colors + good_colors
        for color, agent in zip(colors, self.agents):
            agent.color = color

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)
        # calculate and store distances between all entities
        if self.cache_dists:
            self.calculate_distances()


    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = (agent.mass * agent.accel if agent.accel is not None else agent.mass) * agent.action.u + noise
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_entity_collision_force(a, b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
            if entity_a.movable:
                for wall in self.walls:
                    if wall.bouncy:
                        wf = self.get_wall_collision_force(entity_a, wall)
                        if wf is not None:
                            if p_force[a] is None:
                                p_force[a] = 0.0
                            p_force[a] = p_force[a] + wf
                for line in self.lines:
                    self.apply_line_entity_elastic_collision(entity_a, line)  # Not very clean.. find a better way to integrate entity-line elastic collision than just resetting entity's speed
                    if line.elasticity != 0. and line.exceeds_max_length():
                        if entity_a is line.entity1:
                            p_force[a] += line.elasticity * (line.entity2.state.p_pos - entity_a.state.p_pos)
                        elif entity_a is line.entity2:
                            p_force[a] += line.elasticity * (line.entity1.state.p_pos - entity_a.state.p_pos)

        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for line in self.lines:
            if type(line) is ConnectionLine:
                line.save_previous_endpoints()
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - entity.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.sum(np.square(entity.state.p_vel)))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.max_speed * entity.state.p_vel / speed

            old_pos = np.copy(entity.state.p_pos)
            entity.state.p_pos += entity.state.p_vel * self.dt

            # Per-entity clipping box
            if entity.clip_positions is not None:
                entity.state.p_pos = np.clip(entity.state.p_pos,
                                             a_min=entity.clip_positions[0],
                                             a_max=entity.clip_positions[1])

            # Wall-collision clipping
            for wall in self.walls:
                if not wall.bouncy:
                    clipping_coord = self.get_wall_clipping_coord(wall, entity, old_pos)
                    if clipping_coord is not None:
                        entity.state.p_pos = np.clip(entity.state.p_pos,
                                                     a_min=clipping_coord[0],
                                                     a_max=clipping_coord[1])

        for line in self.lines:
            if type(line) is ConnectionLine:
                if not(line.is_movable()):
                    line.entity1.state.p_pos = line.previous_endpoint1
                    line.entity2.state.p_pos = line.previous_endpoint2

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    # get collision forces for any contact between two entities
    def get_entity_collision_force(self, ia, ib):
        entity_a = self.entities[ia]
        entity_b = self.entities[ib]
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (not entity_a.movable) and (not entity_b.movable):
            return [None, None] # neither entity moves
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        if self.cache_dists:
            delta_pos = self.cached_dist_vect[ia, ib]
            dist = self.cached_dist_mag[ia, ib]
            dist_min = self.min_dists[ia, ib]
        else:
            # compute actual distance between entities
            delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            # minimum allowable distance
            dist_min = entity_a.size + entity_b.size
        # softmax penetration
        dist = dist if abs(dist) > 0.001 else 0.001
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        if entity_a.movable and entity_b.movable:
            # consider mass in collisions
            force_ratio = entity_b.mass / entity_a.mass
            force_a = force_ratio * force
            force_b = -(1 / force_ratio) * force
        else:
            force_a = +force if entity_a.movable else None
            force_b = -force if entity_b.movable else None
        return [force_a, force_b]

    # get collision forces for contact between an entity and a wall
    def get_wall_collision_force(self, entity, wall):
        assert wall.bouncy
        if entity.ghost or not wall.hard:
            return None  # ghost passes through soft walls
        if wall.orient == 'H':
            prll_dim = 0
            perp_dim = 1
        else:
            prll_dim = 1
            perp_dim = 0
        ent_pos = entity.state.p_pos
        if (ent_pos[prll_dim] < wall.endpoints[0] - entity.size or
            ent_pos[prll_dim] > wall.endpoints[1] + entity.size):
            return None  # entity is beyond endpoints of wall
        elif (ent_pos[prll_dim] < wall.endpoints[0] or
              ent_pos[prll_dim] > wall.endpoints[1]):
            # part of entity is beyond wall
            if ent_pos[prll_dim] < wall.endpoints[0]:
                dist_past_end = ent_pos[prll_dim] - wall.endpoints[0]
            else:
                dist_past_end = ent_pos[prll_dim] - wall.endpoints[1]
            theta = np.arcsin(dist_past_end / entity.size)
            dist_min = np.cos(theta) * entity.size + 0.5 * wall.width
        else:  # entire entity lies within bounds of wall
            theta = 0
            dist_past_end = 0
            dist_min = entity.size + 0.5 * wall.width

        # only need to calculate distance in relevant dim
        delta_pos = ent_pos[perp_dim] - wall.axis_pos
        dist = np.abs(delta_pos)
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force_mag = self.contact_force * delta_pos / dist * penetration
        force = np.zeros(2)
        force[perp_dim] = np.cos(theta) * force_mag
        force[prll_dim] = np.sin(theta) * np.abs(force_mag)
        return force

    def get_wall_clipping_coord(self, wall, entity, entity_old_pos):
        assert not wall.bouncy

        if entity.ghost or not wall.hard:
            return None

        if wall.orient == 'H':
            radius_x = abs((max(wall.endpoints) - min(wall.endpoints)) / 2.)
            radius_y = wall.width / 2.
            rect = Rectangle(center=[max(wall.endpoints) - radius_x, wall.axis_pos],
                             r_x=radius_x + entity.size,
                             r_y=radius_y + entity.size)
        elif wall.orient == 'V':
            radius_x = wall.width / 2.
            radius_y = abs((max(wall.endpoints) - min(wall.endpoints)) / 2.)
            rect = Rectangle(center=[wall.axis_pos, max(wall.endpoints) - radius_y],
                             r_x=radius_x + entity.size,
                             r_y=radius_y + entity.size)
        else:
            raise ValueError

        is_inside = rect.is_inside(point=entity.state.p_pos)
        was_inside = rect.is_inside(point=entity_old_pos)

        clipping_coord = None
        if all(is_inside.values()):
            # comes from top
            if not was_inside['top']:
                clipping_coord = np.array([
                    [-np.inf, rect.center[1] + rect.radiuses[1]],
                    [np.inf, np.inf]
                ])
            # comes from bottom
            elif not was_inside['bottom']:
                clipping_coord = np.array([
                    [-np.inf, -np.inf],
                    [np.inf, rect.center[1] - rect.radiuses[1]]
                ])
            # comes from right
            elif not was_inside['right']:
                clipping_coord = np.array([
                    [rect.center[0] + rect.radiuses[0], -np.inf],
                    [np.inf, np.inf]
                ])
            # comes from left
            elif not was_inside['left']:
                clipping_coord = np.array([
                    [-np.inf, -np.inf],
                    [rect.center[0] - rect.radiuses[0], np.inf]
                ])

            return clipping_coord

    # get collision forces for contact between an entity and a line
    def apply_line_entity_elastic_collision(self, entity, line):
        # checks if entity can enter in collision with line
        if not line.hard or line.entity1 is entity or line.entity2 is entity:
            return

        # checks if entity has entered in collision with line
        norm = lambda x: np.sqrt(np.sum(np.square(x)))
        dist1 = norm(entity.state.p_pos - line.start)
        dist2 = norm(entity.state.p_pos - line.end)
        line_length = norm(line.start - line.end)

        if dist1 + dist2 <= line_length + 0.005:

            entity_speed = np.sqrt(np.sum(np.square(entity.state.p_vel)))
            R_i = entity.state.p_vel / (entity_speed + 1e-5)
            R_r = R_i - 2 * line.normal * np.dot(R_i, line.normal)
            entity.state.p_vel = entity_speed * R_r
            entity.has_bounced = True
            # First collision "breaks" the line (the line is not hard anymore)
            line.hard = False
            return
        else:
            return

class Rectangle(object):
    def __init__(self, center, r_x, r_y):

        self.center = np.array(center)
        self.radiuses = np.array([r_x, r_y], dtype=np.float)

    def is_inside(self, point):

        inside = {'top': False, 'bottom': False, 'right': False, 'left':False}

        if point[0] > self.center[0] - self.radiuses[0]:
            inside['left'] = True
        if point[0] < self.center[0] + self.radiuses[0]:
            inside['right'] = True
        if point[1] > self.center[1] - self.radiuses[1]:
            inside['bottom'] = True
        if point[1] < self.center[1] + self.radiuses[1]:
            inside['top'] = True

        return inside