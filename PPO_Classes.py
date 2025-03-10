import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, dropout_rate=0.0):
        super().__init__()

        ## Hidden dimensions and layers can be adjusted as needed based on our state and action spaces

        ## These are the layers for the policy network
        ## state_dim is the number of input features for the current state
        ## action_dim is the number of potential actions, where the model outputs a probability distribution
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, action_dim)

        ## Used the tanh function as discussed in the meeting (and the original paper)
        self.activation = nn.tanh()
        self.dropout = nn.Dropout(dropout_rate) ## Might not be necessary. Defaulted it to 0.0.
        self.softmax = nn.functional.Softmax(dim = -1)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)

        x = self.activation(self.fc2(x))
        x = self.dropout(x)

        x = self.activation(self.fc3(x))
        x = self.dropout(x)

        x = self.activation(self.fc4(x))
        x = self.dropout(x)

        x = self.softmax(self.fc5(x))
        return x
    
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, output_dim = 1, dropout_rate=0.0):
        super().__init__()

        ## Hidden dimensions and layers can be adjusted as needed based on our state space

        ## These are the layers for the policy network
        ## state_dim is the number of input features for the current state
        ## output_dim is 1 for the value function, where it learns the value using the bellman equation

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, output_dim)

        ## Used the tanh function as discussed in the meeting (and the original paper)
        self.activation = nn.tanh()
        self.dropout = nn.Dropout(dropout_rate) ## Might not be necessary. Defaulted it to 0.0.

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)

        x = self.activation(self.fc2(x))
        x = self.dropout(x)

        x = self.activation(self.fc3(x))
        x = self.dropout(x)

        x = self.activation(self.fc4(x))
        x = self.dropout(x)

        x = self.fc5(x)
        return x