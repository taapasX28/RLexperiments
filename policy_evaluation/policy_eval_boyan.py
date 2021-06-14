from typing import Sequence
import Boyan
import numpy as np
from numpy.core.fromnumeric import shape
import tiles3 as tc
from tqdm import tqdm


def planning(n, theta, F, b, gamma, alpha, feature_encoder):
    state_choice = [0,1,2,3,4,5,6,7,8,9,10,11,12]
    sample_state = np.random.choice(state_choice, n)
    for i in range(n):
        phi = feature_encoder.encode(sample_state[i])
        phi_prime = F @ phi
        reward = b.T @ phi
        delta = reward + (gamma*(theta.T @ phi_prime)) - (theta.T @ phi)
        theta += alpha*delta*phi
    return theta

if __name__ == "__main__":

    theta = np.random.uniform(-0.001, 0, size=(4))
    F = np.zeros((4,4))
    b = np.zeros((4))
    alpha = 0.01
    gamma = 1
    epsilon = 0.1
    N_0 = 1000.0
    numEpisodes = 100
    stepsPerEpisode = 12
    rewardTracker = []
    render = False
    solved = False
    n = 10
    loss = []
    env = Boyan.Boyan()
    feature_encoder = Boyan.BoyanRep()


    for episodeNum in tqdm(range(1,numEpisodes+1)):
        G = 0
        state = env.start()
        #print(episodeNum, ":\n")
        for step in range(stepsPerEpisode):
            phi = feature_encoder.encode(state)
            action = 0#policy(state)
            reward, state2 ,done = env.step(action)
            phi_prime = feature_encoder.encode(state2)
                   
            G += reward
            delta = reward + (gamma*(theta.T @ phi_prime)) - (theta.T @ phi)
            #modelling
            F = F + alpha*np.outer((phi_prime-np.dot(F, phi)), phi)
            b = b + alpha*((reward - b.T@phi)*phi)
            theta += alpha*delta*phi
            #plan
            theta = planning(n, theta, F, b, gamma, alpha, feature_encoder)
            state = state2
        alpha = alpha* ((N_0 +1)/(N_0 + (episodeNum)**1.1))
        loss.append(delta**2)

    import matplotlib.pyplot as plt
    plt.plot(loss)
    plt.ylabel('Loss')
    plt.xlabel('Episodes')
    plt.show()
