from numpy.random import chisquare
import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import wrappers
import pickle

pos_space_min = int(np.round(-1.2* 2**6))
pos_space_max =  int(np.round(0.6* 2**6))
vel_space_min = int(np.round(-0.07* 2**8))
vel_space_max =  int(np.round(0.07* 2**8))

def get_state(observation):
    pos, vel =  observation
    pos_bin = int(np.round(pos* 2**6))
    vel_bin = int(np.round(vel * 2**8))
    return (pos_bin, vel_bin)

def max_action(Q, state, actions=[0, 1, 2]):
    values = np.array([Q[state,a] for a in actions])
    action = np.argmax(values)

    return action

def add(s, a, s_prime,r, Model, reward_model):
    Model[s,a] = s_prime
    reward_model[s,a] = r
    return (Model, reward_model)

def sample(visited):
    import random
    vis_list = list(visited.keys())
    sample_state = random.sample(vis_list,1)[0]
    if len(visited[sample_state])>1:
        sample_action = random.sample(visited[sample_state],1)[0]
    else:
        sample_action =  visited[sample_state][0]
    
    return sample_state, sample_action

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 1000
    n_games = 50000
    alpha = 0.1
    gamma = 0.99
    eps = 1.0
    p = 5
    action_space = [0, 1, 2]

    states = []
    for pos in range(pos_space_min, pos_space_max):
        for vel in range(vel_space_min, vel_space_max):
            states.append((pos, vel))

    Q = {}
    for state in states:
        for action in action_space:
            Q[state, action] = 0
    #modelling
    Model = {}
    for state in states:
        for action in action_space:
            Model[state, action] = 0

    reward_model = {}
    for state in states:
        for action in action_space:
            reward_model[state, action] = 0
    #capture visited states
    visited = {}

    score = 0
    total_rewards = np.zeros(n_games)
    for i in range(n_games):
        done = False
        obs = env.reset()
        state = get_state(obs)
        if i % 100 == 0 and i > 0:
            print('episode ', i, 'score ', score, 'epsilon %.3f' % eps)
        score = 0
        while not done:
            action = np.random.choice([0,1,2]) if np.random.random() < eps \
                    else max_action(Q, state)
            obs_, reward, done, info = env.step(action)
            state_ = get_state(obs_)
            score += reward
            action_ = max_action(Q, state_)
            Q[state, action] = Q[state, action] + \
                    alpha*(reward + gamma*Q[state_, action_] - Q[state, action])

            Model, reward_model = add(state, action, state_, reward, Model, reward_model)

            if state in visited.keys():
                if len(visited[state])<3:
                    visited[state].append(action)
                else:
                    pass
            else:
                visited[state] = [action] 

            #planning
            for _ in range(p):
                sam_state, sam_action = sample(visited)
                s_prime = Model[sam_state, sam_action]
                r = reward_model[sam_state, sam_action]
                Q[sam_state, sam_action] = Q[sam_state, sam_action] + \
                    alpha*(r + gamma*Q[s_prime, sam_action] - Q[sam_state, sam_action])

            state = state_
        total_rewards[i] = score
        eps = eps - 2/n_games if eps > 0.01 else 0.01

    mean_rewards = np.zeros(n_games)
    for t in range(n_games):
        mean_rewards[t] = np.mean(total_rewards[max(0, t-50):(t+1)])
    plt.plot(mean_rewards)
    plt.savefig('mountaincar1.png')
