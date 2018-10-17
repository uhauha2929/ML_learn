import random
from collections import namedtuple, deque

import gym
import numpy as np
import torch
import torch.nn as nn

# Hyper Parameters for DQN
GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.5  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 64  # size of mini batch
HIDDEN_SIZE = 10
GRAD_CLIP = 0.5
# Hyper Parameters
EPISODE = 1000  # Episode limitation
STEP = 300  # Step limitation in an episode
TEST = 5  # The number of experiment test every 100 episode

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class Agent(object):
    def __init__(self, env):
        self.env = env
        self.replay_buffer = deque(maxlen=REPLAY_SIZE)
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.steps_done = 0
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, self.action_dim),
        ).to(DEVICE)
        self.criterion = nn.MSELoss().to(DEVICE)
        self.optimizer = torch.optim.Adam(self.net.parameters())

    def egreedy_action(self, state):
        self.steps_done += 1
        state_input = torch.Tensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_value = self.net(state_input).view(-1).cpu().numpy()
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
        return np.argmax(q_value) if random.random() < self.epsilon \
            else np.random.randint(0, self.action_dim)

    def action(self, state):
        state_input = torch.Tensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action_value = self.net(state_input).view(-1).cpu().numpy()
        return np.argmax(action_value)

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append(Transition(state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > BATCH_SIZE:
            self._train()

    def _train(self):
        samples = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch, action_batch, next_state_batch = [], [], []
        for t in samples:
            state_batch.append(torch.Tensor(t.state))
            action_batch.append(torch.Tensor(t.action))
            next_state_batch.append(torch.Tensor(t.next_state))
        state_batch = torch.stack(state_batch).to(DEVICE)
        action_batch = torch.stack(action_batch).to(DEVICE)
        next_state_batch = torch.stack(next_state_batch).to(DEVICE)

        next_q_value = self.net(next_state_batch)
        y_batch = []
        for i in range(0, BATCH_SIZE):
            done = samples[i].done
            if done:
                y_batch.append(samples[i].reward)
            else:
                y_batch.append(samples[i].reward + GAMMA * torch.max(next_q_value[i]))
        y_batch = torch.Tensor(y_batch).to(DEVICE)
        q_value = self.net(state_batch)
        self.optimizer.zero_grad()
        loss = self.criterion(torch.sum(q_value * action_batch, -1), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), GRAD_CLIP)
        self.optimizer.step()


def main():
    env = gym.make('CartPole-v0')
    agent = Agent(env)
    for episode in range(EPISODE):
        # initialize task
        state = env.reset()
        # Train
        for step in range(STEP):
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            # Define reward for agent
            reward = -1 if done else 0.1
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        # Test every 100 episodes
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = agent.action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)


if __name__ == '__main__':
    main()
