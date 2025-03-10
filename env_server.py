from PPO import PPO

## This is where the main environment to server configurations will be
## Change these values based on the environment parameters
state_dim = 5
action_dim = 5
ppo = PPO(state_dim, action_dim)

while True:
    ## With this implementation, current_reward is the reward JUST received
    ## The offsets are taken care of in the PPO code

    ## Pseudocode
    ## 1. Get state features from environment
    ## 2. action = ppo.action(state_features, current_reward)
    ## 3. Perform action in the environment