from torch import nn as nn


class QNetwork(nn.Module):
    """
    Q-Network for descrete action space receives a state and outputs values for each action
    """

    def __init__(self, action_size, state_encoder, hidden_size):
        super().__init__()
        self.state_encoder = state_encoder
        if type(hidden_size) == int:
            fc_layers = [
                nn.Linear(state_encoder.output_size, hidden_size),
                nn.ReLU(inplace=False),
                nn.Linear(hidden_size, action_size)
            ]
        elif type(hidden_size) == list:
            fc_layers = []
            input_size = state_encoder.output_size
            for hs in hidden_size:
                fc_layers.append(nn.Linear(input_size, hs))
                fc_layers.append(nn.ReLU(inplace=False))
                input_size = hs
            fc_layers.append(nn.Linear(input_size, action_size))
        else:
            AttributeError(f"unknown type of {hidden_size} parameter")

        self.fc = nn.Sequential(*fc_layers)
        self.output_size = action_size

    def forward(self, states):
        states_encoding = self.state_encoder(states)
        a = self.fc(states_encoding)
        return a
