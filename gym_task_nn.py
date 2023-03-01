import gym
import numpy as np
from cma import CMAEvolutionStrategy
from network import Network

class GymTaskNN:
    def init(self, env_name, num_hidden_layers, neurons_per_hidden_layer, population_size):
        self.env_name = env_name
        self.num_hidden_layers = num_hidden_layers
        self.neurons_per_hidden_layer = neurons_per_hidden_layer
        self.population_size = population_size
        self.networks = [Network(self.get_num_inputs(), self.get_num_outputs(), num_hidden_layers, neurons_per_hidden_layer) for _ in range(population_size)]
        self.env = gym.make(env_name)
        
    def get_num_inputs(self):
        return self.env.observation_space.shape[0]

    def get_num_outputs(self):
        return self.env.action_space.n

    def get_rewards(self, weights):
        self.networks[0].set_weights(weights)
        state = self.env.reset()
        done = False
        total_reward = 0
        while not done:
            action = np.argmax(self.networks[0].predict(state.reshape(1, -1)))
            state, reward, done, info = self.env.step(action)
            total_reward += reward
        return total_reward

    def train(self, num_iterations):
        es = CMAEvolutionStrategy(self.networks[0].get_weights(), 0.1)
        for i in range(num_iterations):
            solutions = es.ask()
            rewards = [self.get_rewards(weights) for weights in solutions]
            es.tell(solutions, rewards)
            best_weights = es.best.get()[0]
            self.networks[0].set_weights(best_weights)
            print("Iteration {}: Best Reward = {}".format(i+1, self.get_rewards(best_weights)))

    def test(self, num_episodes):
        total_reward = 0
        for i in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = np.argmax(self.networks[0].predict(state.reshape(1, -1)))
                state, reward, done, info = self.env.step(action)
                total_reward += reward
        print("Average Reward over {} episodes: {}".format(num_episodes, total_reward/num_episodes))
