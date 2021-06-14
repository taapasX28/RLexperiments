from typing import Sequence
import mountaincar
import numpy as np
from numpy.core.fromnumeric import shape
import tiles3 as tc
from tqdm import tqdm

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
        #self.n = self.num_tilings
        self.n = iht_size
    
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

        POSITION_MIN = -1.2
        POSITION_MAX = 0.5
        VELOCITY_MIN = -0.07
        VELOCITY_MAX = 0.07

        position_scale = self.num_tiles / (POSITION_MAX - POSITION_MIN)
        velocity_scale = self.num_tiles / (VELOCITY_MAX - VELOCITY_MIN)

        tiles = tc.tiles(self.iht, self.num_tilings, [position * position_scale, 
                                                      velocity * velocity_scale])
        
        return np.array(tiles)
    
    def get_one_hot(self, tiles):
        one_hot = np.zeros(self.n)
        for i in tiles:
            one_hot[i] = 1.0
        return one_hot


def planning(n, theta, F, b, tile, gamma, alpha):
    POSITION_MIN = -1.2
    POSITION_MAX = 0.5
    VELOCITY_MIN = -0.07
    VELOCITY_MAX = 0.07
    sample_pos = np.random.uniform(low=POSITION_MIN, high=POSITION_MAX, size=(n,))
    sample_vel = np.random.uniform(low=VELOCITY_MIN, high=VELOCITY_MAX, size=(n,))
    for i in range(n):
        phi = tile.get_tiles(position = sample_pos[i], velocity = sample_vel[i])
        phi = tile.get_one_hot(phi)
        phi_prime = F @ phi
        reward = b.T @ phi
        delta = reward + (gamma*(theta.T @ phi_prime)) - (theta.T @ phi)
        theta += alpha*delta*phi
    return theta

def policy(state, epsilon):
    velocity = state[1]
    random = np.random.choice([0,1], 1, p=[1-epsilon, epsilon])
    if not random:
        if velocity>=0:
            action = 2
        else:
            action = 1
    else:
        action  = np.random.choice([0,1,2], 1)[0]
    
    return action

if __name__ == "__main__":

    tile = MountainCarTileCoder(iht_size=1000, num_tilings=10, num_tiles=8)
    theta = np.random.uniform(-0.001, 0, size=(tile.n))
    F = np.zeros((tile.n,tile.n))
    b = np.zeros((tile.n))
    alpha = 0.01
    gamma = 0.995
    epsilon = 0.1
    N_0 = 1000000.0
    numEpisodes = 1000
    stepsPerEpisode = 200
    rewardTracker = []
    render = False
    solved = False
    n = 10
    loss = []
    env = mountaincar.MountainCar()


    for episodeNum in tqdm(range(1,numEpisodes+1)):
        G = 0
        env.init()
        state = env.start()
        #print(episodeNum, ":\n")
        for step in range(stepsPerEpisode):
            phi = tile.get_tiles(position=state[0], velocity=state[1])
            phi = tile.get_one_hot(phi)
            action = policy(state, epsilon)    
            reward, state2 ,done = env.step(action)
            phi_prime = tile.get_tiles(position=state2[0], velocity=state2[1])
            phi_prime = tile.get_one_hot(phi_prime)
                   
            G += reward
            delta = reward + (gamma*(theta.T @ phi_prime)) - (theta.T @ phi)
            #modelling
            F = F + alpha*np.outer((phi_prime-np.dot(F, phi)), phi)
            b = b + alpha*((reward - b.T@phi)*phi)
            theta += alpha*delta*phi
            #plan
            theta = planning(n, theta, F, b, tile, gamma, alpha)
            state = state2
        alpha = alpha* ((N_0 +1)/(N_0 + (episodeNum)**1.1))
        loss.append(delta**2)

    import matplotlib.pyplot as plt
    plt.plot(loss)
    plt.ylabel('Loss')
    plt.show()
