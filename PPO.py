import PPO_Functions as F
import PPO_Classes as C
import torch
import torch.optim as optim

class PPO:
    def __init__(self, trajectory_size, state_dim, action_dim):
        ## These are Replay Buffers
        self.old_log_probs = [] ## This is meant to hold the old log probs so we can calculate the policy ratio as a batch
        self.states = []
        self.actions = []
        self.rewards = []

        self.batch_size = trajectory_size
        self.policy_network = C.PolicyNetwork(state_dim, action_dim)
        self.value_network = C.ValueNetwork(state_dim, 1)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=3e-4)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=3e-4)

    ## This function is for each time the agent queries the action to take
    def action(self, state, reward):
        
        ## Rewards are appended before to account for offset by 1
        ## Thought process is that the current reward corresponds to the previous state action pair
        self.rewards.append(reward)

        if len(self.states) == self.batch_size:
            ## Updates policy every batch_size timesteps and clears the lists
            self.update_policy() 

            self.old_log_probs.clear()
            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
        
        action, log_prob = F.get_action(self.policy_network, state)

        ## This part just accumulates the information until the policy update happens
        ## It will reset each policy update
        self.old_log_probs.append(log_prob)
        self.states.append(state)
        self.actions.append(action)

        return action

    def update_policy(self):
        ## Reward will be offset by 1
        ## Represents the current reward is for the previous state/action pair
        loss, value_loss = F.update_policy(
            self.states, self.actions, self.old_log_probs, self.rewards[1:], self.policy_network,
            self.value_network, self.policy_optimizer, self.value_optimizer
        )


        print("MODELS UPDATED", "POLICY LOSS:", loss, "VALUE LOSS", value_loss)
        




