import torch
import random
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

def ppo_loss(old_log_prob, new_log_prob, advantage, epsilon=0.2):
    ratio = torch.exp(new_log_prob - old_log_prob)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    return -torch.min(ratio * advantage, clipped_ratio * advantage).mean()

def get_action(policy_network, state):
    state = torch.tensor(state, dtype=torch.float32)
    action_probs = policy_network(state)
    m = torch.distributions.Categorical(action_probs)
    action = m.sample()
    log_prob = m.log_prob(action)

    return action.item(), log_prob

def adv_function(rewards, values, gamma=0.99, lam=0.95):
    advantages = [0] * len(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        next_value = values[t+1] if t+1 < len(values) else 0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    return advantages


def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def update_policy(
        states, actions, old_log_probs, rewards, policy_network, value_network,
        policy_optimizer, value_optimizer, train_epochs=4, mini_batch_size=64
):
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64)
    old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)

    values = value_network(states).squeeze()
    advantages = adv_function(rewards, values.detach())
    advantages = torch.tensor(advantages, dtype=torch.float32)
    
    dataset = TensorDataset(states, actions, old_log_probs, advantages)
    data_loader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)
    
    for _ in range(train_epochs):
        for mini_states, mini_actions, mini_old_log_probs, mini_advantages in data_loader:
            action_probs = policy_network(mini_states)
            m = torch.distributions.Categorical(action_probs)
            new_log_probs = m.log_prob(mini_actions)
            loss = ppo_loss(mini_old_log_probs, new_log_probs, mini_advantages)
            
            policy_optimizer.zero_grad()
            loss.backward()
            policy_optimizer.step()
    
    returns = compute_returns(rewards, gamma=0.99)
    returns = torch.tensor(returns, dtype=torch.float32)
    values = value_network(states).squeeze()
    value_loss = torch.nn.functional.mse_loss(values, returns)
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()
    
    return loss.item(), value_loss.item()
