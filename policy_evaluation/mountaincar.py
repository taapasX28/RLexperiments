import numpy as np

class MountainCar():

    actions = [0, 1, 2]

    def __init__(self):
        reward = 0.0
        observation = 0
        termination = False
        self.current_state = None
        self.reward_obs_term = (reward, observation, termination)
        self.count = 0

    actions = [0, 1, 2]

    def __init__(self):
        reward = 0.0
        observation = 0
        termination = False
        self.current_state = None
        self.reward_obs_term = (reward, observation, termination)
        self.count = 0

    def init(self):
        local_observation = 0  # An empty NumPy array

        self.reward_obs_term = (0.0, local_observation, False)


    def start(self):

        position = np.random.uniform(-0.6, -0.4)
        velocity = 0.0
        self.current_state = np.array([position, velocity]) # position, velocity

        return self.current_state

    def step(self, action):
        position, velocity = self.current_state

        terminal = False
        reward = -1.0
        velocity = self.bound_velocity(velocity + 0.001 * (action - 1) - 0.0025 * np.cos(3 * position))
        position = self.bound_position(position + velocity)

        if position == -1.2:
            velocity = 0.0
        elif position == 0.5:
            self.current_state = None
            terminal = True
            reward = 0.0

        self.current_state = np.array([position, velocity])

        self.reward_obs_term = (reward, self.current_state, terminal)

        return self.reward_obs_term


    def bound_velocity(self, velocity):
        if velocity > 0.07:
            return 0.07
        if velocity < -0.07:
            return -0.07
        return velocity

    def bound_position(self, position):
        if position > 0.5:
            return 0.5
        if position < -1.2:
            return -1.2
        return position
