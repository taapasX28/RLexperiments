from typing import Sequence
import gym
import numpy as np
from numpy.core.fromnumeric import shape
env = gym.make("MountainCar-v0")
import time
import tiles3 as tc


class MountainCarTileCoder:
    def __init__(self, iht_size=4096, num_tilings=8, num_tiles=8):
        """
        Initializes the MountainCar Tile Coder
        Initializers:
        iht_size -- int, the size of the index hash table, typically a power of 2
        num_tilings -- int, the number of tilings
        num_tiles -- int, the number of tiles. Here both the width and height of the
                     tile coder are the same
        Class Variables:
        self.iht -- tc.IHT, the index hash table that the tile coder will use
        self.num_tilings -- int, the number of tilings the tile coder will use
        self.num_tiles -- int, the number of tiles the tile coder will use
        """
        self.iht = tc.IHT(iht_size)
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        self.actions = env.action_space.n
        self.n = num_tilings
    
    def get_tiles(self, position, velocity):
        """
        Takes in a position and velocity from the mountaincar environment
        and returns a numpy array of active tiles.
        
        Arguments:
        position -- float, the position of the agent between -1.2 and 0.5
        velocity -- float, the velocity of the agent between -0.07 and 0.07
        returns:
        tiles - np.array, active tiles
        """
        # Set the max and min of position and velocity to scale the input
        # POSITION_MIN
        # POSITION_MAX
        # VELOCITY_MIN
        # VELOCITY_MAX
        ### START CODE HERE ###
        POSITION_MIN = env.observation_space.low[0]
        POSITION_MAX = env.observation_space.high[0]
        VELOCITY_MIN = env.observation_space.low[1]
        VELOCITY_MAX = env.observation_space.high[1]
        ### END CODE HERE ###
        
        # Use the ranges above and self.num_tiles to set position_scale and velocity_scale
        # position_scale = number of tiles / position range
        # velocity_scale = number of tiles / velocity range
        
        # Scale position and velocity by multiplying the inputs of each by their scale
        
        ### START CODE HERE ###
        position_scale = self.num_tiles / (POSITION_MAX - POSITION_MIN)
        velocity_scale = self.num_tiles / (VELOCITY_MAX - VELOCITY_MIN)
        ### END CODE HERE ###
        
        # get the tiles using tc.tiles, with self.iht, self.num_tilings and [scaled position, scaled velocity]
        # nothing to implment here
        tiles = tc.tiles(self.iht, self.num_tilings, [position * position_scale, 
                                                      velocity * velocity_scale])
        
        return np.array(tiles)

    def getQval(self, phi, F, reward, gamma, theta):
        Q = max(min(-200, reward + gamma * (theta.T @ F) @ phi),0)
        return Q

    def getQ(self, phi, F, rew, gamma, theta):
        Q = np.zeros(self.actions)
        for i in range(self.actions):
            Q[i] = self.getQval(phi, F[i], rew[i], gamma, theta)
        return Q
        
def argmax(q_values):
    top = float("-inf")
    ties = []

    for i in range(len(q_values)):
        if q_values[i] > top:
            top = q_values[i]
            ties = []

        if q_values[i] == top:
            ties.append(i)

    return np.random.choice(ties)

def rewardlist(state, tile):
    rew = []
    env2 = gym.make("MountainCar-v0")
    env2.reset()
    for i in range(tile.actions):
        env2.state =state
        state2, reward, done, info = env2.step(i)
        rew.append(reward)
    return rew

def planning(n, theta, F, tile, gamma, alpha):
    for _ in range(n):
        sample_state = env.observation_space.sample()
        phi = tile.get_tiles(position = sample_state[0], velocity = sample_state[1])
        rew = rewardlist(state, tile)
        Q = tile.getQ( phi, F, rew, gamma, theta)
        delta = max(Q) - (theta.T @ phi)
        theta += alpha*delta*phi
    return theta

if __name__ == "__main__":

    tile = MountainCarTileCoder(iht_size=1024, num_tilings=8, num_tiles=8)
    theta = np.random.uniform(-0.001, 0, size=(tile.n))
    F = 3* [np.zeros((tile.n,tile.n))]
    alpha = 0.000001
    gamma = 0.995
    numEpisodes = 100000
    stepsPerEpisode = 1000
    rewardTracker = []
    render = False
    solved = False
    n = 10

    for episodeNum in range(1,numEpisodes+1):
        G = 0
        state = env.reset()
        for step in range(stepsPerEpisode):
            if render:
                env.render()
            phi = tile.get_tiles(position=state[0], velocity=state[1])    
            rew = rewardlist(state, tile)
            Q = tile.getQ( phi, F, rew, gamma, theta)
            action = argmax(Q)
            state2, reward, done, info = env.step(action)
            phi_prime = tile.get_tiles(position=state2[0], velocity=state2[1])                    
            G += reward
            delta = reward + (gamma*(theta.T @ phi_prime)) - (theta.T @ phi)
            if done == True:
                theta += alpha*delta*phi
                rewardTracker.append(G)
                if episodeNum %100 == 0:
                    print('Total Episodes = {} Episode Reward = {} Average reward = {}'.format(episodeNum, G, np.mean(rewardTracker)))
                break
            #modelling
            F[action] = F[action] + alpha*np.outer((phi_prime-np.dot(F[action], phi)), phi)
            #b[action] = b[action] + alpha*((reward - b[action].T@phi)*phi)
            
            theta += alpha*delta*phi
            #plan
            #uncomment for planning
            theta = planning(n, theta, F, tile, gamma, alpha)
            state = state2

        if solved != True:
            if episodeNum > 100:
                if sum(rewardTracker[episodeNum-100:episodeNum])/100 >= -110:
                    print('Solve in {} Episodes'.format(episodeNum))
                    render = True
                    solved = True
