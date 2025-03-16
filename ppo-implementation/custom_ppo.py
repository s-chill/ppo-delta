from typing import Dict, List
import gymnasium as gym
import time

import numpy as np
import time
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import torch.nn.functional as F


if torch.cuda.is_available():
    torch.set_default_device("cuda")
elif torch.backends.mps.is_available():  # For Apple Metal (M1/M2 chips)
    torch.set_default_device("mps")
elif torch.backends.opencl.is_available():  # OpenCL support
    torch.set_default_device("opencl")
else:
    torch.set_default_device("cpu")

print("Torch is using device:", torch.get_default_device())


class PPO:
	def __init__(self, env, **hyperparameters):
		self._init_hyperparameters(hyperparameters)

		self.env = env
		self.obs_dim = env.observation_space.shape[0]
		self.act_dim = env.action_space.shape[0]

		self.actor = ActorCriticNetwork(self.obs_dim, self.act_dim)
		self.critic = ActorCriticNetwork(self.obs_dim, 1)

		self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

		self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
		self.cov_mat = torch.diag(self.cov_var)

		self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,
			'i_so_far': 0,
			'batch_lens': [],
			'batch_rews': [],
			'actor_losses': [],
		}

		self.metadata: List = []

	def learn(self, total_timesteps):
		print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
		print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
		t_so_far = 0
		i_so_far = 0
		while t_so_far < total_timesteps:
			batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

			t_so_far += np.sum(batch_lens)

			i_so_far += 1

			self.logger['t_so_far'] = t_so_far
			self.logger['i_so_far'] = i_so_far

			V, _ = self.evaluate(batch_obs, batch_acts)
			A_k = batch_rtgs - V.detach()

			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

			for _ in range(self.n_updates_per_iteration):
				V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

				ratios = torch.exp(curr_log_probs - batch_log_probs)

				surr1 = ratios * A_k
				surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

				actor_loss = (-torch.min(surr1, surr2)).mean()
				critic_loss = nn.MSELoss()(V, batch_rtgs)

				self.actor_optim.zero_grad()
				actor_loss.backward(retain_graph=True)
				self.actor_optim.step()

				self.critic_optim.zero_grad()
				critic_loss.backward()
				self.critic_optim.step()

				self.logger['actor_losses'].append(actor_loss.detach())

			self.store_data()

			if i_so_far % self.save_freq == 0:
				torch.save(self.actor.state_dict(), self.actor_save_path)
				torch.save(self.critic.state_dict(), self.critic_save_path)
				print(f"Saved models to {self.actor_save_path} and {self.critic_save_path}")

		return self.metadata

	def rollout(self):
		batch_obs = []
		batch_acts = []
		batch_log_probs = []
		batch_rews = []
		batch_rtgs = []
		batch_lens = []

		ep_rews = []

		t = 0

		while t < self.timesteps_per_batch:
			ep_rews = []

			obs, _ = self.env.reset()
			done = False

			for ep_t in range(self.max_timesteps_per_episode):
				if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
					self.env.render()

				t += 1

				batch_obs.append(obs)
				action, log_prob = self.get_action(obs)
				obs, rew, terminated, truncated, _ = self.env.step(action)
				ep_rews.append(rew)
				batch_acts.append(action)
				batch_log_probs.append(log_prob)
				if terminated or truncated:
					break

			batch_lens.append(ep_t + 1)
			batch_rews.append(ep_rews)

		batch_obs = torch.tensor(batch_obs, dtype=torch.float)
		batch_acts = torch.tensor(batch_acts, dtype=torch.float)
		batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
		batch_rtgs = self.compute_rtgs(batch_rews)

		self.logger['batch_rews'] = batch_rews
		self.logger['batch_lens'] = batch_lens

		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

	def compute_rtgs(self, batch_rews):
		batch_rtgs = []

		for ep_rews in reversed(batch_rews):

			discounted_reward = 0

			for rew in reversed(ep_rews):
				discounted_reward = rew + discounted_reward * self.gamma
				batch_rtgs.insert(0, discounted_reward)

		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

		return batch_rtgs

	def get_action(self, obs):
		mean = self.actor(obs)

		dist = MultivariateNormal(mean, self.cov_mat)

		action = dist.sample()

		log_prob = dist.log_prob(action)

		return action.detach().numpy(), log_prob.detach()

	def evaluate(self, batch_obs, batch_acts):
		V = self.critic(batch_obs).squeeze()

		mean = self.actor(batch_obs)
		dist = MultivariateNormal(mean, self.cov_mat)
		log_probs = dist.log_prob(batch_acts)

		return V, log_probs

	def _init_hyperparameters(self, hyperparameters):
		self.timesteps_per_batch = 4800
		self.max_timesteps_per_episode = 1600
		self.n_updates_per_iteration = 5
		self.lr = 0.005
		self.gamma = 0.95
		self.clip = 0.2

		self.actor_save_path = './ppo_actor.pth'
		self.critic_save_path = './ppo_critic.pth'

		self.render = True
		self.render_every_i = 10
		self.save_freq = 10
		self.seed = None

		for param, val in hyperparameters.items():
			exec('self.' + param + ' = ' + str(val))

		if self.seed != None:
			assert(type(self.seed) == int)

			torch.manual_seed(self.seed)
			print(f"Successfully set seed to {self.seed}")

	def store_data(self):
		iteration = self.logger['i_so_far']
		average_episode_rewards = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
		avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])
		timesteps = self.logger['t_so_far']

		self.metadata.append({
			'iteration': iteration,
			'average_episode_rewards': average_episode_rewards,
			'avg_actor_loss': avg_actor_loss,
			'timesteps': timesteps,
		})

		self.logger['batch_lens'] = []
		self.logger['batch_rews'] = []
		self.logger['actor_losses'] = []


class ActorCriticNetwork(nn.Module):
	def __init__(self, in_dim, out_dim):
		super(ActorCriticNetwork, self).__init__()

		self.layer1 = nn.Linear(in_dim, 64)
		self.layer2 = nn.Linear(64, 64)
		self.layer3 = nn.Linear(64, out_dim)

	def forward(self, obs):
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)

		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		output = self.layer3(activation2)

		return output