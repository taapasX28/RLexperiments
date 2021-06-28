# Policy evaluation by Random Linear Dyna Algorithm
# Created by Taapas Agrawal
# 28-06-2021

from typing import Sequence
import Boyan
import numpy as np
from numpy.core.fromnumeric import shape
import tiles3 as tc
from tqdm import tqdm


def planning(n, theta, F, b, gamma, alpha, feature_encoder):
    #planning steps
    state_choice = [i for i in range(98)]
    np.random.seed(0)
    for i in range(n):
        sample_state = np.random.choice(state_choice)
        phi = feature_encoder.encode(sample_state)
        phi_prime = F @ phi
        reward = b.T @ phi
        delta = reward + (gamma*(theta.T @ phi_prime)) - (theta.T @ phi)
        theta += alpha*delta*phi
    return theta

def policy(state):
    """Uniform random policy for Boyan chain env"""
    if state<=2:
        action = 0
    else:
        action = np.random.choice([0,1])
    return action

def get_val(env, gamma):
    """Gets true analytical value of states by solving
    Bellman equation directly"""
    P, R = env.getPR()
    I = np.identity(98)
    value_states = R @ np.linalg.inv((I-gamma*P))
    return value_states

if __name__ == "__main__":

    theta = np.random.uniform(-0.001, 0, size=(25))
    F = np.zeros((25,25))
    b = np.zeros((25))
    alpha = 0.01
    gamma = 1
    epsilon = 0.1
    N_0 = 100.0
    numEpisodes = 100
    stepsPerEpisode = 100
    rewardTracker = []
    n = 150
    loss = []
    env = Boyan.Boyan()
    feature_encoder = Boyan.BoyanRep()
    value_states = get_val(env, gamma)
    map = feature_encoder.getmap()

    for episodeNum in tqdm(range(1,numEpisodes+1)):
        G = 0
        state = env.start()
        for step in range(stepsPerEpisode):
            phi = feature_encoder.encode(state)
            action = policy(state)
            reward, state2 ,done = env.step(action)
            phi_prime = feature_encoder.encode(state2)
            if done == True:
                break          
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
        L = np.linalg.norm(value_states - np.dot(theta ,map.T))
        loss.append(L)

    import matplotlib.pyplot as plt
    plt.plot(loss)
    plt.ylabel('Loss')
    plt.xlabel('Episodes')
    plt.show()
