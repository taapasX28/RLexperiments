import gym
import numpy as np
env = gym.make("MountainCar-v0")
import time

class tilecoder:
	
    def __init__(self, numTilings, tilesPerTiling):
        self.maxIn = env.observation_space.high
        self.minIn = env.observation_space.low
        self.numTilings = numTilings
        self.tilesPerTiling = tilesPerTiling
        self.dim = len(self.maxIn)
        self.numTiles = (self.tilesPerTiling**self.dim) * self.numTilings
        self.actions = env.action_space.n
        self.n = self.numTiles
        self.tileSize = np.divide(np.subtract(self.maxIn,self.minIn), self.tilesPerTiling-1)
		
    def getFeatures(self, variables):
        ### ENSURES LOWEST POSSIBLE INPUT IS ALWAYS 0
        self.variables = np.subtract(variables, self.minIn)
        tileIndices = np.zeros(self.numTilings)
        matrix = np.zeros([self.numTilings,self.dim])
        for i in range(self.numTilings):
            for i2 in range(self.dim):
                matrix[i,i2] = int(self.variables[i2] / self.tileSize[i2] \
                    + i / self.numTilings)
        for i in range(1,self.dim):
            matrix[:,i] *= self.tilesPerTiling**i
        for i in range(self.numTilings):
            tileIndices[i] = (i * (self.tilesPerTiling**self.dim) \
                + sum(matrix[i,:])) 
        return tileIndices

    def oneHotVector(self, features):
        oneHot = np.zeros((self.n,1))
        for i in features:
            index = int(i)
            oneHot[index] = 1
        return oneHot

    def getQval(self, phi, F, b, gamma, theta):
        Q = b.T @ phi + gamma * (theta.T @ F @ phi)
        return Q

    def getQ(self, phi, F, b, gamma, theta):
        Q = np.zeros(self.actions)
        for i in range(self.actions):
            Q[i] = self.getQval(phi, F[i], b[i], gamma, theta)
        return Q


def planning(n, theta, F, b, tile, gamma, alpha):
    for _ in range(n):
        sample_state = env.observation_space.sample()
        action = env.action_space.sample()
        phi = tile.getFeatures(sample_state)
        phi = tile.oneHotVector(phi)
        phi_prime  = F[action] @ phi
        reward = (b[action].T @ phi)
        delta = reward + (gamma*(theta.T @ phi_prime)) - (theta.T @ phi)
        theta += alpha*delta*phi
        return theta



if __name__ == "__main__":

    tile = tilecoder(4,18)
    
    theta = np.random.uniform(-0.001, 0, size=(tile.n,1))
    F = 3* [np.random.uniform(-0.001, 0, size=(tile.n, tile.n))]
    b =  3* [np.random.uniform(-0.001, 0, size=(tile.n,1))]
    alpha = .1/ tile.numTilings*3.2
    gamma = 1
    numEpisodes = 100000
    stepsPerEpisode = 200
    rewardTracker = []
    render = False
    solved = False
    n = 50

    for episodeNum in range(1,numEpisodes+1):
        G = 0
        state = env.reset()
        for step in range(stepsPerEpisode):
            if render:
                env.render()
            phi = tile.getFeatures(state)
            phi = tile.oneHotVector(phi)        
            Q = tile.getQ( phi, F, b, gamma, theta)
            action = np.argmax(Q)
            state2, reward, done, info = env.step(action)
            phi_prime = tile.getFeatures(state2)
            phi_prime = tile.oneHotVector(phi_prime)                    
            G += reward
            delta = reward + (gamma*(theta.T @ phi_prime)) - (theta.T @ phi)
            if done == True:
                theta += alpha*delta*phi
                rewardTracker.append(G)
                if episodeNum %100 == 0:
                    print('Total Episodes = {} Episode Reward = {} Average reward = {}'.format(episodeNum, G, np.mean(rewardTracker)))
                break
            #modelling
            F[action] += alpha*((phi_prime -F[action]@phi)@ phi.T)
            b[action] += alpha*((reward - b[action].T@phi)*phi)
            theta += alpha*delta*phi
            #plan
            #uncomment for planning
            #theta = planning(n, theta, F, b, tile, gamma, alpha)
            state = state2

        if solved != True:
            if episodeNum > 100:
                if sum(rewardTracker[episodeNum-100:episodeNum])/100 >= -110:
                    print('Solve in {} Episodes'.format(episodeNum))
                    render = True
                    solved = True
