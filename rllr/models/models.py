import torch
from torch import nn

from ..utils import convert_to_torch


class WorkerNetwork(nn.Module):
    def __init__(self, state_encoder, emb_size, action_size, config):
        super().__init__()
        self.state_encoder = state_encoder
        hidden_size = config['head.hidden_size']
        if type(hidden_size) == int:
            fc_layers = [
                nn.Linear(state_encoder.output_size + emb_size, hidden_size),
                nn.ReLU(inplace=False),
                nn.Linear(hidden_size, action_size)
            ]
        elif type(hidden_size) == list:
            fc_layers = []
            input_size = state_encoder.output_size + emb_size
            for hs in hidden_size:
                fc_layers.append(nn.Linear(input_size, hs))
                fc_layers.append(nn.ReLU(inplace=False))
                input_size = hs
            fc_layers.append(nn.Linear(input_size, action_size))
        else:
            AttributeError(f"unknown type of {hidden_size} parameter")

        self.fc = nn.Sequential(*fc_layers)
        self.output_size = action_size

    def forward(self, states, goal_states_emb):
        states_emb = self.state_encoder(states)
        x = torch.cat((goal_states_emb, states_emb), 1)
        return self.fc(x)


class MasterNetwork(nn.Module):

    """
    Master network. Input is current state and output is state embedding.
    """
    def __init__(self, emb_size, goal_state_encoder, config):
        super().__init__()
        self.state_encoder = goal_state_encoder
        hidden_size = config['head.hidden_size']
        input_size = goal_state_encoder.output_size
        fc_layers = []
        if type(hidden_size) == int:
            fc_layers += [
                nn.Linear(input_size, hidden_size),
                nn.ReLU(inplace=False),
                nn.Linear(hidden_size, emb_size)
            ]
        elif type(hidden_size) == list:
            for hs in hidden_size:
                fc_layers.append(nn.Linear(input_size, hs))
                fc_layers.append(nn.ReLU(inplace=False))
                input_size = hs
            fc_layers.append(nn.Linear(input_size, emb_size))
        else:
            AttributeError(f"unknown type of {hidden_size} parameter")
        fc_layers.append(nn.Tanh())
        self.fc = nn.Sequential(*fc_layers)
        self.output_size = emb_size

    def forward(self, states):
        x = self.state_encoder(states)
        return self.fc(x)


class MasterWorkerNetwork(nn.Module):
    """
    Master-Worker model
    """
    def __init__(self, master, worker):
        super(MasterWorkerNetwork, self).__init__()
        self.master = master
        self.worker = worker

    def forward(self, states):
        if 'goal_emb' in states.keys():
            goal_states_emb = states['goal_emb']
        else:
            goal_states_emb = self.master(states['goal_state'])
        return self.worker(states['state'], goal_states_emb)


def get_master_worker_net(state_encoder, goal_state_encoder, action_size, config):
    worker = WorkerNetwork(
        state_encoder=state_encoder,
        emb_size=goal_state_encoder.output_size,
        action_size=action_size,
        config=config['worker']
    )

    master = MasterNetwork(
        emb_size=goal_state_encoder.output_size,
        goal_state_encoder=goal_state_encoder,
        config=config['master']
    )
    return MasterWorkerNetwork(master=master, worker=worker)


class StateDistanceNetwork(nn.Module):
    """
    State-Embedding model
    """
    def __init__(self, encoder, action_size, config):
        super().__init__()
        self.encoder = encoder

        self.fc = nn.Sequential(
            nn.Linear(2 * encoder.output_size, config['hidden_size']),
            nn.ReLU(inplace=False),
            nn.Linear(config['hidden_size'], action_size),
            nn.LogSoftmax(dim=-1)
        )
        self.output_size = action_size

    def forward(self, state, next_state):
        x = self.encoder(state)
        y = self.encoder(next_state)

        return self.fc(torch.cat((x, y), 1))


class EncoderDistance:
    def __init__(self, encoder, device, threshold=5):
        self.encoder = encoder.to(device)
        self.device = device
        self.threshold = threshold

    def __call__(self, state, goal_state):
        if isinstance(state, dict):
            state, goal_state = state['image'], goal_state['image']
        with torch.no_grad():
            embeds = self.encoder(convert_to_torch([state, goal_state], device=self.device))
        return torch.dist(embeds[0], embeds[1], 2).cpu().item() < self.threshold
