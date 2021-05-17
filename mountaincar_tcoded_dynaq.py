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
        self.n = self.numTiles * self.actions
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

    def oneHotVector(self, features, action):
        oneHot = np.zeros(self.n)
        for i in features:
            index = int(i + (self.numTiles*action))
            oneHot[index] = 1
        return oneHot

    def getVal(self, theta, features, action):
        val = 0 
        for i in features:
            index = int(i + (self.numTiles*action))
            val += theta[index]
        return val

    def getQ(self, features, theta):
        Q = np.zeros(self.actions)
        for i in range(self.actions):
            Q[i] = tile.getVal(theta, features, i)
        return Q


class Model():
    def __init__(self, n_states, n_actions):
        self.transitions = np.zeros((n_states,n_actions), dtype=np.uint8)
        self.rewards = np.zeros((n_states, n_actions))

    def add(self,s,a,s_prime,r):
        self.transitions[s,a] = s_prime
        self.rewards[s,a] = r

    def sample(self):
        """ Return random state, action"""
        # Random visited state
        s = np.random.choice(np.where(np.sum(self.transitions, axis=1) > 0)[0])
        # Random action in that state
        a = np.random.choice(np.where(self.transitions[s] > 0)[0])
        return s,a

    def step(self, s,a):
        """ Return state_prime and reward for state-action pair"""
        s_prime = self.transitions[s,a]
        r = self.rewards[s,a]
        return s_prime, r

def planning(n, theta, tile, gamma, alpha):
    for i in range(n):
        state, action =  self.model.sample()
        s_prime, reward = self.model.step(state, action)
        F = tile.getFeatures(state)
        Q = tile.getQ(F, theta)
        delta = reward - Q[action]
        Q = tile.getQ(tile.getFeatures(s_prime), theta)
        delta += gamma*np.max(Q)
        theta += np.multiply((alpha*delta), tile.oneHotVector(F,action))
    return theta


if __name__ == "__main__":

    tile = tilecoder(4,18)
    theta = np.random.uniform(-0.001, 0, size=(tile.n))
    alpha = .1/ tile.numTilings*3.2
    gamma = 1
    numEpisodes = 100000
    stepsPerEpisode = 200
    rewardTracker = []
    render = False
    solved = False
    n = 50
    model =  Model(env.observation_space.n, env.action_space.n)

    for episodeNum in range(1,numEpisodes+1):
        G = 0
        state = env.reset()
        for step in range(stepsPerEpisode):
            if render:
                env.render()
            F = tile.getFeatures(state)
            Q = tile.getQ(F, theta)
            action = np.argmax(Q)
            state2, reward, done, info = env.step(action)
            G += reward
            delta = reward - Q[action]
            if done == True:
                theta += np.multiply((alpha*delta), tile.oneHotVector(F,action))
                rewardTracker.append(G)
                if episodeNum %100 == 0:
                    print('Total Episodes = {} Episode Reward = {} Average reward = {}'.format(episodeNum, G, np.mean(rewardTracker)))
                break
            Q = tile.getQ(tile.getFeatures(state2), theta)
            delta += gamma*np.max(Q)
            theta += np.multiply((alpha*delta), tile.oneHotVector(F,action))

            #learn model
            model.add(state, action, state2, reward)
            #plan
            theta = planning(n, theta, tile, gamma, alpha)

            state = state2

        if solved != True:
            if episodeNum > 100:
                if sum(rewardTracker[episodeNum-100:episodeNum])/100 >= -110:
                    print('Solve in {} Episodes'.format(episodeNum))
                    render = True
                    solved = True
